"""Telemetry through the full standard pipeline (CPU baseline, real sample video).

SYNTHETIC TELEMETRY: the sample video came without a flight log. The matching track below is
made from the job's own camera path through a known similarity (10 m per unit, a rotation and
an offset), so the test checks that the pipeline recovers it and labels the result correctly.
It says nothing about real GNSS accuracy. The hover track checks the rejection path.
"""

import math
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_jobs import app_client, start_job, worker_client  # noqa: F401 - fixture re-export
from test_missions import upload
from test_standard_pipeline import SAMPLE, sample_mission

from nazar_contracts import PipelineMode
from nazar_worker.glb import read_glb
from nazar_worker.las import read_las
from nazar_worker.runner import run_loop

M_PER_DEG_LAT = 111_070.0
LAT0, LON0 = 41.8902, 12.4922
SCALE = 10.0  # sample1 camera path spans ~3.3 units: ~33 m, above the 20 m minimum

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="sample video not present")


def run_job(client: TestClient, mission_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    job = start_job(client, mission_id, pipeline="standard")
    run_loop(worker_client(client), "w1", [PipelineMode.STANDARD], once=True)
    snap: dict[str, Any] = client.get(f"/api/jobs/{job['id']}").json()
    assert snap["status"] == "completed", snap["error"]
    by_kind = {a["kind"]: a for a in snap["artifacts"]}
    report = client.get(f"/api/artifacts/{by_kind['report']['id']}/content").json()
    return snap, report


def keyframe_track(client: TestClient, snap: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Video times and local camera centres of the placed keyframes."""
    traj = next(a for a in snap["artifacts"] if a["kind"] == "camera_trajectory")
    poses = client.get(f"/api/artifacts/{traj['id']}/content").json()["poses"]
    times = np.array([p["time_s"] for p in poses])
    return times, np.array([p["position"] for p in poses])


def attach(client: TestClient, mission_id: str, t: np.ndarray, enu: np.ndarray) -> None:
    lat = LAT0 + enu[:, 1] / M_PER_DEG_LAT
    lon = LON0 + enu[:, 0] / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
    rows = ["time_s,latitude,longitude,rel_alt_m"] + [
        f"{a:.3f},{b:.8f},{c:.8f},{d:.3f}"
        for a, b, c, d in zip(t, lat, lon, enu[:, 2], strict=True)
    ]
    data = "\n".join(rows).encode()
    asset = client.post(
        f"/api/missions/{mission_id}/assets",
        json={"kind": "telemetry", "filename": "synthetic.csv", "byte_size": len(data)},
    ).json()
    assert upload(client, asset["id"], data)["status"] == "ready"


def test_matching_telemetry_gives_metric_scale(
    app_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAZAR_FAST3R_PYTHON", "")
    mission = sample_mission(app_client)
    first, report = run_job(app_client, mission)
    assert "sensor_optimization" in report["skipped"]
    times, centres = keyframe_track(app_client, first)
    # Local (+Y up) -> ENU: a proper rotation, SCALE metres per unit, 40 m above take-off.
    c, s = math.cos(0.6), math.sin(0.6)
    rot = np.array([[c, 0.0, -s], [s, 0.0, c], [0.0, 1.0, 0.0]])
    dense_t = np.arange(times.min(), times.max() + 0.05, 0.2)
    enu_kf = SCALE * centres @ rot.T + np.array([5.0, -8.0, 40.0])
    dense = np.stack([np.interp(dense_t, times, enu_kf[:, i]) for i in range(3)], axis=1)
    attach(app_client, mission, dense_t, dense)

    snap, report = run_job(app_client, mission)
    al = report["sensor_alignment"]
    assert al["scale_accepted"], al["rejected_because"]
    # sample1's camera moves along a short, nearly straight arc: the rotation about that line
    # is undetermined, so the pipeline accepts the scale but refuses to georeference. (The
    # ENU branch is covered by tests/test_georef.py in the worker.)
    assert not al["orientation_accepted"]
    assert al["planarity"] < 0.15 and "straight line" in al["rejected_because"][0]
    assert "not_georeferenced" in {w["code"] for w in snap["warnings"]}
    assert math.isclose(al["metres_per_unit"], SCALE, rel_tol=0.05)
    assert al["residual_rms_m"] < 5
    assert report["input"]["telemetry"] is True
    assert "sensor_optimization" not in report["skipped"]
    assert report["coordinate_frame"]["scale_status"] == "metric_sensor"
    cloud = next(a for a in snap["artifacts"] if a["kind"] == "point_cloud")
    assert cloud["frame"]["scale_status"] == "metric_sensor"
    assert math.isclose(cloud["frame"]["metres_per_unit"], al["metres_per_unit"], rel_tol=1e-4)
    assert "scale_from_gnss" in {w["code"] for w in snap["warnings"]}
    assert "scale_unknown" not in {w["code"] for w in snap["warnings"]}
    export = next(a for a in snap["artifacts"] if a["media_type"] == "application/vnd.las")
    las = read_las(app_client.get(f"/api/artifacts/{export['id']}/content").content)
    assert las.system_id == "NAZAR local frame m, no CRS"
    # Metres: the exported cloud's extent is SCALE times its local extent.
    local = np.array(cloud["bounds"]["max"]) - np.array(cloud["bounds"]["min"])
    las_extent = np.asarray(las.bounds_max) - np.asarray(las.bounds_min)
    assert math.isclose(
        float(np.linalg.norm(las_extent)), SCALE * float(np.linalg.norm(local)), rel_tol=0.05
    )
    glb = next(a for a in snap["artifacts"] if a["media_type"] == "model/gltf-binary")
    scene = read_glb(app_client.get(f"/api/artifacts/{glb['id']}/content").content)
    assert scene.doc["asset"]["extras"]["units"] == "metre"
    glb_extent = scene.points.max(axis=0) - scene.points.min(axis=0)
    assert math.isclose(
        float(np.linalg.norm(glb_extent)), SCALE * float(np.linalg.norm(local)), rel_tol=0.05
    )


def test_hover_telemetry_is_reported_but_not_applied(
    app_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAZAR_FAST3R_PYTHON", "")
    mission = sample_mission(app_client)
    t = np.arange(0, 60, 0.5)
    attach(app_client, mission, t, np.tile([0.0, 0.0, 30.0], (len(t), 1)))
    snap, report = run_job(app_client, mission)
    al = report["sensor_alignment"]
    assert not al["scale_accepted"]
    assert any("at least 20 m" in r for r in al["rejected_because"])
    assert report["coordinate_frame"]["scale_status"] == "unknown"
    warning = next(w for w in snap["warnings"] if w["code"] == "scale_unknown")
    assert "not applied" in warning["message"]
    stage = next(s for s in snap["stages"] if s["stage"] == "sensor_optimization")
    assert stage["state"] == "completed"
