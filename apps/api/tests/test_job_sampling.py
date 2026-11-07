# ruff: noqa: F811
"""REAL DATA: choosing the frame rate for a job (sample1, real preflight and FFmpeg)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_jobs import app_client, worker_client  # noqa: F401 - fixture re-export
from test_standard_pipeline import SAMPLE, sample_mission

from nazar_contracts import (
    AUTO_MAX_FRAMES,
    AUTO_MIN_FRAMES,
    MAX_FRAMES,
    ComputeTarget,
    PipelineMode,
)

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="sample video not present")


def create(client: TestClient, mission: str, **body: Any) -> Any:
    return client.post(f"/api/missions/{mission}/jobs", json={"pipeline": "standard", **body})


def test_default_is_automatic_and_never_starves_a_short_clip(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    job = create(app_client, mission).json()
    s = job["sampling"]
    assert s["auto"] is True
    assert s["frames"] >= AUTO_MIN_FRAMES - 2  # a window with no usable frame may be dropped
    assert s["frames"] <= AUTO_MAX_FRAMES[ComputeTarget.LOCAL]
    assert s["clamped"] is False
    assert s["cap"] == MAX_FRAMES[ComputeTarget.LOCAL]


def test_an_explicit_rate_is_not_automatic(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    s = create(app_client, mission, fps=1.0).json()["sampling"]
    assert s["auto"] is False
    assert s["fps_requested"] == 1.0
    assert abs(s["fps_effective"] - 1.0) < 0.35


def test_a_higher_rate_gives_more_frames(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    one = create(app_client, mission, fps=1.0).json()
    app_client.post(f"/api/jobs/{one['id']}/cancel")
    two = create(app_client, mission, fps=2.0).json()
    assert two["sampling"]["frames"] > one["sampling"]["frames"] * 1.5


def test_the_rate_is_validated(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    assert create(app_client, mission, fps=0.1).status_code == 422
    assert create(app_client, mission, fps=9).status_code == 422


def test_a_rate_too_low_for_the_video_is_refused_with_advice(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    response = create(app_client, mission, fps=0.25)
    if response.status_code == 201:  # a long clip can still give three frames at 0.25 fps
        assert response.json()["sampling"]["frames"] >= 3
    else:
        assert response.status_code == 422
        assert "higher rate" in response.json()["detail"]


def test_the_worker_gets_the_sampled_frames_not_the_upload_keyframes(
    app_client: TestClient,
) -> None:
    mission = sample_mission(app_client)
    job = create(app_client, mission, fps=1.0).json()
    worker = worker_client(app_client)
    lease = worker.lease("w1", [PipelineMode.STANDARD])
    assert lease is not None
    assert lease.sampling is not None
    assert lease.keyframe_count == job["sampling"]["frames"]
    report = worker.preflight(lease.job_id)
    assert len(report["keyframes"]) == job["sampling"]["frames"]
    assert {k["reason"] for k in report["keyframes"]} == {"sampled"}
    assert worker.keyframe(lease.job_id, 1)[:2] == b"\xff\xd8"  # a real JPEG


def test_the_test_pipeline_ignores_the_rate(app_client: TestClient) -> None:
    mission = sample_mission(app_client)
    job = app_client.post(f"/api/missions/{mission}/jobs", json={"pipeline": "test", "fps": 3})
    assert job.status_code == 201
    assert job.json()["sampling"] is None
