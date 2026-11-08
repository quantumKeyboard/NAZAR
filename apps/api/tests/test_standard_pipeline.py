"""REAL DATA: the standard pipeline on the supplied sample video, through the real API and the
real worker runner in-process. The CPU baseline always runs; the GPU pipeline (Fast3R plus the
other models) runs only where its runtime is installed (`var/gpu` or NAZAR_*_PYTHON), otherwise
it is skipped."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from test_jobs import app_client, start_job, worker_client  # noqa: F401 - fixture re-export

from nazar_api.models import AssetRecord
from nazar_api.storage import PREFLIGHT_DIR, asset_dir
from nazar_contracts import PipelineMode
from nazar_media import run_preflight
from nazar_worker.fast3r import Fast3rRuntime
from nazar_worker.glb import read_glb
from nazar_worker.las import read_las, to_las_axes
from nazar_worker.ply import ESTIMATED, OBSERVED, read_confidence, read_ply
from nazar_worker.runner import run_loop

SAMPLE = Path(__file__).resolve().parents[3] / "docs" / "sample" / "video" / "sample1.mp4"
FAST3R = Fast3rRuntime.from_env()


def sample_mission(client: TestClient) -> str:
    """A mission whose sample video went through the real preflight (real keyframes)."""
    mission = client.post("/api/missions", json={"name": "Sample"}).json()
    settings = client.app.state.settings  # type: ignore[attr-defined]
    with Session(client.app.state.engine) as db:  # type: ignore[attr-defined]
        row = AssetRecord(
            mission_id=mission["id"],
            kind="video",
            filename="sample1.mp4",
            stored_name="source.mp4",
            byte_size=SAMPLE.stat().st_size,
            received_bytes=SAMPLE.stat().st_size,
            status="ready",
            sha256="b" * 64,
        )
        db.add(row)
        db.commit()
        root = asset_dir(settings, mission["id"], row.id)
        folder = root / PREFLIGHT_DIR
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE, root / "source.mp4")  # jobs sample their frames from the video
    run_preflight(SAMPLE, folder, asset_id=row.id, source_sha256="b" * 64, max_duration_s=600)
    return str(mission["id"])


def run_standard(client: TestClient) -> tuple[dict[str, Any], dict[str, Any]]:
    job = start_job(client, sample_mission(client), pipeline="standard")
    run_loop(worker_client(client), "w1", [PipelineMode.STANDARD], once=True)
    snap: dict[str, Any] = client.get(f"/api/jobs/{job['id']}").json()
    assert snap["status"] == "completed", snap["error"]
    by_kind = {a["kind"]: a for a in snap["artifacts"]}
    report = client.get(f"/api/artifacts/{by_kind['report']['id']}/content").json()
    return snap, report


def disable_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CPU-only path even where the GPU environments are installed."""
    for var in ("NAZAR_FAST3R_PYTHON", "NAZAR_GPU_PYTHON", "NAZAR_SPLAT_PYTHON"):
        monkeypatch.setenv(var, "")


def cloud_points(client: TestClient, snap: dict[str, Any], report: dict[str, Any]) -> int:
    """Point count of the final cloud, checking that its per-point confidence classes match
    the report's counts."""
    mpu = report["coordinate_frame"]["metres_per_unit"]
    cloud = next(a for a in snap["artifacts"] if a["kind"] == "point_cloud")
    data = client.get(f"/api/artifacts/{cloud['id']}/content").content
    pts = read_ply(data)
    assert len(pts) == cloud["point_count"]
    labels = read_confidence(data)
    assert labels is not None
    assert int((labels == OBSERVED).sum()) == report["confidence"]["observed"]
    assert int((labels == ESTIMATED).sum()) == report["confidence"]["estimated"]
    # The LAS export holds the same points (axes converted to +Z up), colours and classes.
    export = next(a for a in snap["artifacts"] if a["media_type"] == "application/vnd.las")
    assert export["media_type"] == "application/vnd.las"
    las = read_las(client.get(f"/api/artifacts/{export['id']}/content").content)
    xyz = np.array([p[:3] for p in pts], dtype=np.float64)
    assert np.abs(las.xyz - to_las_axes(xyz * (mpu or 1.0))).max() < 1e-3
    assert (las.user_data == labels).all()
    if mpu is None:
        assert las.system_id == "NAZAR local frame, no CRS"
    assert "las" in report["exports"]
    # The GLB holds the same points and colours (glTF shares NAZAR's +Y-up axes) and the path.
    glb = next(
        a
        for a in snap["artifacts"]
        if a["kind"] == "export" and a["media_type"] == "model/gltf-binary"
    )
    scene = read_glb(client.get(f"/api/artifacts/{glb['id']}/content").content)
    assert np.abs(scene.points - xyz * (mpu or 1.0)).max() < 1e-3
    assert (scene.colors == np.array([p[3:6] for p in pts], dtype=np.uint8)).all()
    assert scene.path is not None and len(scene.path) >= 2
    assert scene.doc["asset"]["extras"]["units"] == ("metre" if mpu else "unitless")
    return len(pts)


pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="sample video not present")


def test_cpu_baseline_reconstructs_the_sample(
    app_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disable_gpu(monkeypatch)
    snap, report = run_standard(app_client)
    assert report["geometry_method"] == "sfm"
    assert report["geometry"]["registered_keyframes"] >= 18
    assert report["geometry"]["mean_reproj_px"] < 1.5
    assert report["geometry"]["points"] == cloud_points(app_client, snap, report)
    assert {a["provenance"]["producer"] for a in snap["artifacts"]} == {"nazar-worker/sfm-baseline"}
    assert "fast3r_fallback" not in {w["code"] for w in snap["warnings"]}


@pytest.mark.gpu
@pytest.mark.skipif(not isinstance(FAST3R, Fast3rRuntime), reason=f"{FAST3R}")
def test_fast3r_reconstructs_the_sample_densely(app_client: TestClient) -> None:  # noqa: F811
    snap, report = run_standard(app_client)
    geometry = report["geometry"]
    assert report["geometry_method"] == "fast3r", geometry.get("fast3r_fallback_reason")
    assert "fast3r_fallback" not in {w["code"] for w in snap["warnings"]}
    assert geometry["placed_keyframes"] == report["input"]["keyframes"]
    n = cloud_points(app_client, snap, report)
    assert n == geometry["points"] and n > 50_000  # dense, unlike the ~6k-point baseline
    assert report["resources"]["peak_vram_mib"] > 0
    cloud = next(a for a in snap["artifacts"] if a["kind"] == "point_cloud")
    assert cloud["provenance"]["producer"] == "nazar-worker/fast3r"
    # Every model of the local profile contributed or says why it did not.
    for key in ("dynamic_analysis", "depth_priors", "surface_processing", "gaussian_splat"):
        assert key in report["contributions"], key
    assert report["resources"]["within_gpu_budget"] is not False
    traj = next(a for a in snap["artifacts"] if a["kind"] == "camera_trajectory")
    poses = app_client.get(f"/api/artifacts/{traj['id']}/content").json()["poses"]
    assert len(poses) == report["input"]["keyframes"]
    # Cameras fly above the levelled ground: median camera height above the median point.
    heights = sorted(p["position"][1] for p in poses)
    assert heights[len(heights) // 2] > 0
    assert report["confidence"]["observed"] > 0 and report["confidence"]["estimated"] > 0


def test_fast3r_failure_falls_back_to_the_baseline_visibly(
    app_client: TestClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A configured but broken GPU runtime must not fail the job or hide what happened."""
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"not a checkpoint")
    broken = tmp_path / "python.exe"
    broken.write_text("")  # exists, but is not a runnable interpreter
    disable_gpu(monkeypatch)
    monkeypatch.setenv("NAZAR_FAST3R_PYTHON", str(broken))
    monkeypatch.setenv("NAZAR_FAST3R_WEIGHTS", str(weights))
    snap, report = run_standard(app_client)
    assert report["geometry_method"] == "sfm"
    assert "fast3r_fallback_reason" in report["geometry"]
    warning = next(w for w in snap["warnings"] if w["code"] == "fast3r_fallback")
    assert "CPU baseline" in warning["message"]
    assert all(
        a["provenance"]["producer"] == "nazar-worker/sfm-baseline" for a in snap["artifacts"]
    )


def test_runtime_is_unavailable_without_configuration() -> None:
    assert isinstance(Fast3rRuntime.from_env({}), str)
    assert "does not exist" in str(
        Fast3rRuntime.from_env({"NAZAR_FAST3R_PYTHON": "nope", "NAZAR_FAST3R_WEIGHTS": "nope"})
    )
