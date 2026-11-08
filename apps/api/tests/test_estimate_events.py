# ruff: noqa: F811
"""The worker's time estimate, stage GPU peaks and model readiness reach the browser API."""

from fastapi.testclient import TestClient
from test_jobs import (  # noqa: F401 - fixture re-export
    WORKER,
    app_client,
    lease,
    post_events,
    ready_mission,
    start_job,
)


def test_estimate_and_stage_peaks_are_in_the_snapshot(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    leased = lease(app_client)
    estimate = {
        "frames": 22,
        "basis": "Built-in estimate.",
        "stages": {"preprocessing": 1.0, "geometry": 30.0},
        "total_s": 31.0,
    }
    response = post_events(
        app_client,
        job["id"],
        leased["lease_id"],
        {"key": "e", "type": "job.estimate", "data": estimate},
        {"key": "s", "type": "stage.started", "stage": "preprocessing"},
        {
            "key": "c",
            "type": "stage.completed",
            "stage": "preprocessing",
            "data": {"duration_s": 1.2, "peak_vram_mib": 4321, "budget_mib": 5632},
        },
    )
    assert response.status_code == 200, response.text
    snap = app_client.get(f"/api/jobs/{job['id']}").json()
    assert snap["estimate"]["total_s"] == 31.0
    assert snap["estimate"]["stages"]["geometry"] == 30.0
    done = next(s for s in snap["stages"] if s["stage"] == "preprocessing")
    assert (done["peak_vram_mib"], done["budget_mib"]) == (4321, 5632)


def test_a_malformed_estimate_is_rejected(app_client: TestClient) -> None:
    job = start_job(app_client, ready_mission(app_client))
    leased = lease(app_client)
    bad = {"key": "e", "type": "job.estimate", "data": {"frames": -1}}
    assert post_events(app_client, job["id"], leased["lease_id"], bad).status_code == 409


def test_reported_models_show_up_in_readiness(app_client: TestClient) -> None:
    models = [
        {"name": "Fast3R", "stage": "geometry", "ready": True, "detail": "Fast3R ViT-L/512"},
        {"name": "YOLO", "stage": "dynamic_analysis", "ready": False, "detail": "not installed"},
    ]
    response = app_client.post(
        "/api/worker/lease",
        json={"worker_id": "w1", "modes": ["standard"], "location": "local", "models": models},
        headers=WORKER,
    )
    assert response.status_code == 204
    local = next(
        w for w in app_client.get("/api/ready").json()["workers"] if w["location"] == "local"
    )
    assert [m["name"] for m in local["models"]] == ["Fast3R", "YOLO"]
    assert local["models"][1]["ready"] is False
