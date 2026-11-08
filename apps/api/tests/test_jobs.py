"""Job lifecycle, worker protocol and event streaming, with the real worker runner talking
to the real app in-process (test pipeline: no GPU, no weights)."""

import hashlib
import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import HEADERS, make_settings
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from nazar_api.jobs import expire_leases
from nazar_api.main import create_app
from nazar_api.models import AssetRecord, JobRecord, utcnow
from nazar_api.storage import PREFLIGHT_DIR, asset_dir
from nazar_contracts import ComputeTarget, GeometryBackend, PipelineMode, StageName, WorkerLease
from nazar_worker.client import WorkerClient
from nazar_worker.ply import read_ply
from nazar_worker.runner import JobRun, geometry_backend, run_loop
from nazar_worker.stages import StageContext

TOKEN = "test-only-worker-token-not-a-secret"
WORKER = {"Authorization": f"Bearer {TOKEN}", **HEADERS}
KEYFRAMES = 5


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[TestClient]:
    settings = make_settings(tmp_path, worker_token=TOKEN)
    with TestClient(create_app(settings), headers=HEADERS) as c:
        response = c.post(
            "/api/auth/setup", json={"email": "ops@example.org", "password": "long enough pw"}
        )
        assert response.status_code == 201
        yield c


def ready_mission(client: TestClient) -> str:
    """A mission whose video already passed validation (skips decoding the sample)."""
    mission = client.post("/api/missions", json={"name": "Jobs"}).json()
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with Session(client.app.state.engine) as db:  # type: ignore[attr-defined]
        row = AssetRecord(
            mission_id=mission["id"],
            kind="video",
            filename="clip.mp4",
            stored_name="source.mp4",
            byte_size=4,
            received_bytes=4,
            status="ready",
            sha256="a" * 64,
        )
        db.add(row)
        db.commit()
        folder = asset_dir(settings, mission["id"], row.id) / PREFLIGHT_DIR
    (folder / "keyframes").mkdir(parents=True)
    for i in range(1, KEYFRAMES + 1):
        (folder / "keyframes" / f"{i:04d}.jpg").write_bytes(b"\xff\xd8fake")
    keyframes = [{"index": 10 * i, "t": i / 2, "image": i} for i in range(1, KEYFRAMES + 1)]
    (folder / "preflight.json").write_text(json.dumps({"keyframes": keyframes}))
    return str(mission["id"])


def start_job(client: TestClient, mission_id: str, pipeline: str = "test") -> dict[str, Any]:
    response = client.post(f"/api/missions/{mission_id}/jobs", json={"pipeline": pipeline})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def worker_client(
    client: TestClient, location: ComputeTarget = ComputeTarget.LOCAL
) -> WorkerClient:
    raw = TestClient(client.app)  # no session cookie; no lifespan re-entry

    def send(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        response = raw.request(
            method, url.removeprefix("http://testserver"), content=body, headers=headers
        )
        return response.status_code, response.content

    return WorkerClient("http://testserver", TOKEN, attempts=1, transport=send, location=location)


def events(client: TestClient, job_id: str, after: int = 0) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = client.get(f"/api/jobs/{job_id}/events?after={after}").json()
    return body


def lease(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/api/worker/lease", json={"worker_id": "w1", "modes": ["test"]}, headers=WORKER
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def post_events(client: TestClient, job_id: str, lease_id: str, *batch: dict[str, Any]) -> Any:
    return client.post(
        f"/api/worker/jobs/{job_id}/events",
        json={"events": list(batch)},
        headers={**WORKER, "X-Nazar-Lease": lease_id},
    )


def test_test_pipeline_runs_end_to_end_through_the_real_worker(app_client: TestClient) -> None:
    mission_id = ready_mission(app_client)
    job = start_job(app_client, mission_id)
    assert job["status"] == "queued" and job["pipeline"] == "test"
    assert [e["type"] for e in events(app_client, job["id"])] == ["job.created", "job.queued"]

    handled = run_loop(worker_client(app_client), "w1", [PipelineMode.TEST], once=True)
    assert handled == 1

    final = app_client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] == "completed", final
    states = {s["stage"]: s["state"] for s in final["stages"]}
    assert states == {
        "preprocessing": "completed",
        "geometry": "completed",
        "depth_priors": "skipped",
        "dynamic_analysis": "skipped",
        "sensor_optimization": "skipped",
        "dense_reconstruction": "completed",
        "surface_processing": "skipped",
        "analysis": "completed",
        "exporting": "completed",
    }
    skipped = {s["stage"]: s["reason"] for s in final["stages"] if s["state"] == "skipped"}
    assert "telemetry" in skipped["sensor_optimization"]
    assert [w["code"] for w in final["warnings"]] == ["test_pipeline"]

    log = events(app_client, job["id"])
    assert [e["seq"] for e in log] == list(range(1, len(log) + 1))
    assert final["last_seq"] == len(log)
    types = [e["type"] for e in log]
    assert types[-1] == "job.completed" and types.count("job.completed") == 1
    assert types.count("preview.updated") == 3  # two previews and the final cloud
    # The first preview arrives before the job completes: progressive, not all-at-once.
    assert types.index("preview.updated") < types.index(
        "stage.completed", types.index("preview.updated")
    )

    by_kind = {a["kind"]: a for a in final["artifacts"]}
    assert set(by_kind) == {"point_cloud_preview", "point_cloud", "camera_trajectory", "report"}
    assert by_kind["point_cloud_preview"]["revision"] == 2
    for manifest in final["artifacts"]:
        assert manifest["provenance"]["test_pipeline"] is True
        assert "path" not in json.dumps(manifest).lower()
    completed = log[-1]["data"]["artifact_ids"]
    assert set(completed) == {
        by_kind[k]["id"] for k in ("point_cloud", "camera_trajectory", "report")
    }

    cloud = by_kind["point_cloud"]
    content = app_client.get(f"/api/artifacts/{cloud['id']}/content")
    assert content.status_code == 200
    assert hashlib.sha256(content.content).hexdigest() == cloud["checksum_sha256"]
    points = read_ply(content.content)
    assert len(points) == cloud["point_count"]
    assert cloud["frame"]["type"] == "local_reconstruction"
    assert cloud["frame"]["scale_status"] == "unknown"

    # Replay from a cursor returns only the tail.
    assert [e["seq"] for e in events(app_client, job["id"], after=len(log) - 2)] == [
        len(log) - 1,
        len(log),
    ]


def test_websocket_replays_from_cursor_and_closes_after_terminal_event(
    app_client: TestClient,
) -> None:
    job = start_job(app_client, ready_mission(app_client))
    run_loop(worker_client(app_client), "w1", [PipelineMode.TEST], once=True)
    total = app_client.get(f"/api/jobs/{job['id']}").json()["last_seq"]

    received: list[dict[str, Any]] = []
    with (
        app_client.websocket_connect(f"/api/jobs/{job['id']}/stream?after=3") as ws,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        while True:
            received.append(ws.receive_json())
    assert closed.value.code == 1000
    assert [e["seq"] for e in received] == list(range(4, total + 1))


def test_websocket_rejects_foreign_origin_and_anonymous_clients(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    path = f"/api/jobs/{job['id']}/stream"
    with (
        pytest.raises(WebSocketDisconnect) as foreign,
        app_client.websocket_connect(path, headers={"origin": "https://evil.example"}),
    ):
        pass
    assert foreign.value.code == 4403
    with (
        pytest.raises(WebSocketDisconnect) as anonymous,
        TestClient(app_client.app).websocket_connect(path),
    ):
        pass
    assert anonymous.value.code == 4404


def test_job_needs_a_ready_video_and_only_one_runs_per_mission(app_client: TestClient) -> None:
    empty = app_client.post("/api/missions", json={"name": "Empty"}).json()
    response = app_client.post(f"/api/missions/{empty['id']}/jobs", json={"pipeline": "test"})
    assert response.status_code == 409
    mission_id = ready_mission(app_client)
    start_job(app_client, mission_id)
    again = app_client.post(f"/api/missions/{mission_id}/jobs", json={"pipeline": "test"})
    assert again.status_code == 409


def test_worker_routes_require_the_worker_token(app_client: TestClient, tmp_path: Path) -> None:
    body = {"worker_id": "w1", "modes": ["test"]}
    assert app_client.post("/api/worker/lease", json=body).status_code == 401
    wrong = {**HEADERS, "Authorization": "Bearer nope"}
    assert app_client.post("/api/worker/lease", json=body, headers=wrong).status_code == 401
    assert app_client.post("/api/worker/lease", json=body, headers=WORKER).status_code == 204

    unconfigured = create_app(make_settings(tmp_path / "other"))
    with TestClient(unconfigured, headers=WORKER) as c:
        assert c.post("/api/worker/lease", json=body).status_code == 503


def test_events_are_idempotent_and_must_follow_the_state_machine(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    got = lease(app_client)
    assert got["job_id"] == job["id"] and got["keyframe_count"] == KEYFRAMES
    lid = got["lease_id"]

    start = {"key": "pre:start", "type": "stage.started", "stage": "preprocessing"}
    assert post_events(app_client, job["id"], lid, start).status_code == 200
    seq = app_client.get(f"/api/jobs/{job['id']}").json()["last_seq"]
    assert post_events(app_client, job["id"], lid, start).json()["last_seq"] == seq  # dedup

    wrong_stage = {
        "key": "x",
        "type": "stage.progress",
        "stage": "geometry",
        "data": {"message": "m"},
    }
    assert post_events(app_client, job["id"], lid, wrong_stage).status_code == 409
    early = {"key": "done", "type": "job.completed"}
    assert post_events(app_client, job["id"], lid, early).status_code == 409
    forged = {"key": "f", "type": "artifact.available", "data": {}}
    assert post_events(app_client, job["id"], lid, forged).status_code == 422
    bad_data = {
        "key": "b",
        "type": "stage.progress",
        "stage": "preprocessing",
        "data": {"total": -1},
    }
    assert post_events(app_client, job["id"], lid, bad_data).status_code == 409
    backwards = {
        "key": "skip",
        "type": "stage.skipped",
        "stage": "preprocessing",
        "data": {"reason": "r"},
    }
    assert post_events(app_client, job["id"], lid, backwards).status_code == 409
    assert app_client.get(f"/api/jobs/{job['id']}").json()["last_seq"] == seq
    assert post_events(app_client, job["id"], "stale-lease", start).status_code == 409


def test_artifact_bytes_must_match_the_declared_checksum(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    lid = lease(app_client)["lease_id"]
    headers = {**WORKER, "X-Nazar-Lease": lid}
    content = b"ply\nnot really\n"
    declared = {
        "key": "a1",
        "kind": "report",
        "media_type": "application/json",
        "byte_size": len(content),
        "checksum_sha256": hashlib.sha256(content).hexdigest(),
        "provenance": {"producer": "test", "test_pipeline": True},
        "final": True,
    }
    created = app_client.post(
        f"/api/worker/jobs/{job['id']}/artifacts", json=declared, headers=headers
    ).json()
    url = f"/api/worker/jobs/{job['id']}/artifacts/{created['id']}/content"
    assert app_client.put(url, content=b"x" * len(content), headers=headers).status_code == 422
    assert app_client.get(f"/api/artifacts/{created['id']}/content").status_code == 404
    assert app_client.put(url, content=content, headers=headers).status_code == 200
    again = app_client.post(
        f"/api/worker/jobs/{job['id']}/artifacts", json=declared, headers=headers
    )
    assert again.json() == {**created, "uploaded": True}
    types = [e["type"] for e in events(app_client, job["id"])]
    assert types.count("artifact.available") == 1


def test_lost_lease_requeues_the_job_and_locks_out_the_old_worker(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    lid = lease(app_client)["lease_id"]
    start = {"key": "s", "type": "stage.started", "stage": "preprocessing"}
    post_events(app_client, job["id"], lid, start)
    with Session(app_client.app.state.engine) as db:  # type: ignore[attr-defined]
        row = db.get(JobRecord, job["id"])
        assert row is not None
        row.lease_expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
        assert expire_leases(db) == 1

    snap = app_client.get(f"/api/jobs/{job['id']}").json()
    assert snap["status"] == "queued" and snap["attempt"] == 2
    assert all(s["state"] == "pending" for s in snap["stages"])
    assert post_events(app_client, job["id"], lid, start).status_code == 409
    assert lease(app_client)["attempt"] == 2


def test_cancel_queued_job_is_immediate_and_running_job_waits_for_worker(
    app_client: TestClient,
) -> None:
    queued = start_job(app_client, ready_mission(app_client))
    done = app_client.post(f"/api/jobs/{queued['id']}/cancel").json()
    assert done["status"] == "cancelled"
    assert app_client.post(f"/api/jobs/{queued['id']}/cancel").status_code == 409

    running = start_job(app_client, ready_mission(app_client))
    lid = lease(app_client)["lease_id"]
    headers = {**WORKER, "X-Nazar-Lease": lid}
    post_events(
        app_client,
        running["id"],
        lid,
        {"key": "s", "type": "stage.started", "stage": "preprocessing"},
    )
    assert (
        app_client.post(f"/api/jobs/{running['id']}/cancel").json()["status"] == "cancel_requested"
    )
    beat = app_client.post(f"/api/worker/jobs/{running['id']}/heartbeat", headers=headers).json()
    assert beat["cancel_requested"] is True
    new_stage = {"key": "g", "type": "stage.started", "stage": "geometry"}
    assert post_events(app_client, running["id"], lid, new_stage).status_code == 409
    post_events(app_client, running["id"], lid, {"key": "c", "type": "job.cancelled"})
    assert app_client.get(f"/api/jobs/{running['id']}").json()["status"] == "cancelled"

    retried = app_client.post(f"/api/jobs/{running['id']}/retry").json()
    assert retried["status"] == "queued" and retried["attempt"] == 2


def test_standard_job_is_refused_when_its_frames_cannot_be_sampled(app_client: TestClient) -> None:
    # The fixture's video is a stub: the frames cannot be read, so no job is created and the
    # reason is plain, instead of a job that queues and fails later on the GPU.
    mission = ready_mission(app_client)
    response = app_client.post(f"/api/missions/{mission}/jobs", json={"pipeline": "standard"})
    assert response.status_code in (422, 500)
    assert "sampled" in response.json()["detail"] or "short" in response.json()["detail"]
    assert app_client.get(f"/api/missions/{mission}/jobs").json() == []


class _CancelBeforeStart:
    """Simulates the user pressing Cancel after the worker's last cancel check but before
    it reports `stage.started` (the window between the two)."""

    def __init__(self, client: TestClient, job_id: str) -> None:
        self.client, self.job_id = client, job_id

    def skip_reason(self, lease: WorkerLease) -> str | None:
        assert self.client.post(f"/api/jobs/{self.job_id}/cancel").status_code == 200
        return None

    def run(self, ctx: StageContext) -> None:
        raise AssertionError("must not run after cancellation")


def test_cancel_racing_a_stage_start_ends_cancelled_not_failed(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    worker = worker_client(app_client)
    lease = worker.lease("w1", [PipelineMode.TEST])
    assert lease is not None
    run = JobRun(
        worker, lease, {StageName.PREPROCESSING: _CancelBeforeStart(app_client, job["id"])}, 60
    )
    assert run.execute() == "cancelled"
    snap = app_client.get(f"/api/jobs/{job['id']}").json()
    assert snap["status"] == "cancelled"
    assert snap["error"] is None


def test_jobs_are_private(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    assert app_client.get("/api/jobs/does-not-exist").status_code == 404
    app_client.post("/api/auth/logout")
    assert app_client.get(f"/api/jobs/{job['id']}").status_code == 401
    assert app_client.get(f"/api/jobs/{job['id']}/events").status_code == 401


def test_readiness_reports_the_worker_geometry_backend(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gpu() -> dict[str, str]:
        checks = app_client.get("/api/ready").json()["checks"]
        return next(c for c in checks if c["name"] == "gpu")

    assert gpu()["status"] == "not_configured"
    assert "No worker" in gpu()["detail"]
    worker = worker_client(app_client)
    monkeypatch.setenv("NAZAR_FAST3R_PYTHON", "")
    assert (
        worker.lease("w1", [PipelineMode.STANDARD], geometry_backend([PipelineMode.STANDARD]))
        is None
    )
    assert gpu()["status"] == "not_configured"
    assert "CPU baseline" in gpu()["detail"]
    assert worker.lease("w1", [PipelineMode.TEST], GeometryBackend.FAST3R) is None
    assert gpu()["status"] == "ok"
    assert geometry_backend([PipelineMode.TEST]) is None
