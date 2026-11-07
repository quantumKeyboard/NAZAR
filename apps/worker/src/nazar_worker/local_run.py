"""Run the whole standard pipeline on one video, in-process, with no API or web app.

    nazar-worker local --video PATH --how local|clg [--fps 1] [--telemetry PATH] [--out DIR]

The same stage adapters as a leased job run against `LocalClient`, which serves the
preflight and keyframes from disk and writes every artifact to `<out>/artifacts`. Model
inputs and outputs (frames, masks, depth, splat renders, mesh) stay in `<out>/work`, so each
model's contribution can be inspected. At the end a summary prints: per-stage time and status,
peak GPU memory per model against the budget, and what each model contributed.

Exit code 0 only when the job completed, every model the profile requires ran, and (on
`local`) every GPU step stayed within the memory budget.
"""

import argparse
import json
import mimetypes
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from nazar_contracts import (
    AUTO_MAX_FRAMES,
    FPS_MAX,
    FPS_MIN,
    MAX_FRAMES,
    ArtifactCreate,
    ArtifactCreated,
    ComputeTarget,
    EventType,
    JobStatus,
    PipelineMode,
    WorkerEvent,
    WorkerEventAck,
    WorkerHeartbeat,
    WorkerLease,
)
from nazar_media import MediaError, run_preflight
from nazar_media.sampling import sample_job_frames
from nazar_media.telemetry import TelemetryError, parse_telemetry, track_to_json
from nazar_worker.fast3r import Fast3rRuntime
from nazar_worker.forecast import Forecast
from nazar_worker.gpu import GpuRuntime
from nazar_worker.profile import PROFILES, ComputeProfile
from nazar_worker.runner import JobRun
from nazar_worker.standard import build_standard_pipeline

EXTENSIONS = {
    "application/vnd.nazar.ply": ".ply",
    "application/vnd.las": ".las",
    "model/gltf-binary": ".glb",
    "application/json": ".json",
    "application/zip": ".zip",
    "application/vnd.nazar.gaussians+ply": ".ply",
}
MODEL_STAGES = (
    "dynamic_analysis",
    "geometry",
    "depth_priors",
    "dense_reconstruction",
    "gaussian_splat",
    "surface_processing",
)


class LocalClient:
    """`WorkerApi` backed by a preflight folder; artifacts land in `out/artifacts`."""

    def __init__(self, preflight_dir: Path, out: Path, telemetry: dict[str, Any] | None) -> None:
        self.preflight_dir = preflight_dir
        self.artifacts = out / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.telemetry_doc = telemetry
        self.events_log: list[dict[str, Any]] = []
        self.published: list[dict[str, Any]] = []
        self._status = JobStatus.QUEUED
        self._t0 = time.monotonic()

    def heartbeat(self, job_id: str) -> WorkerHeartbeat:
        return WorkerHeartbeat(
            cancel_requested=False, lease_expires_at=datetime.now(UTC) + timedelta(hours=1)
        )

    def events(self, job_id: str, events: list[WorkerEvent]) -> WorkerEventAck:
        for event in events:
            record = json.loads(event.model_dump_json())
            record["t"] = round(time.monotonic() - self._t0, 2)
            self.events_log.append(record)
            if event.stage is not None:
                self._status = JobStatus(event.stage.value)
            _print_event(record)
        return WorkerEventAck(
            status=self._status, last_seq=len(self.events_log), cancel_requested=False
        )

    def preflight(self, job_id: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (self.preflight_dir / "preflight.json").read_text(encoding="utf-8")
        )
        return data

    def keyframe(self, job_id: str, image: int) -> bytes:
        return (self.preflight_dir / "keyframes" / f"{image:04d}.jpg").read_bytes()

    def telemetry(self, job_id: str) -> dict[str, Any] | None:
        return self.telemetry_doc

    def publish(self, job_id: str, declared: ArtifactCreate, content: bytes) -> ArtifactCreated:
        ext = (
            EXTENSIONS.get(declared.media_type)
            or mimetypes.guess_extension(declared.media_type)
            or ".bin"
        )
        path = self.artifacts / f"{declared.key}{ext}"
        path.write_bytes(content)
        self.published.append(
            {
                "key": declared.key,
                "kind": declared.kind.value,
                "file": str(path),
                "bytes": len(content),
                "producer": declared.provenance.producer,
            }
        )
        return ArtifactCreated(id=declared.key, revision=1, uploaded=True)


def _print_event(e: dict[str, Any]) -> None:
    stage = e.get("stage") or "job"
    data = e.get("data") or {}
    kind = e["type"]
    prefix = f"[{e['t']:8.1f}s] {stage:21}"
    if kind == EventType.STAGE_PROGRESS.value:
        done, total, unit = data.get("completed"), data.get("total"), data.get("unit") or ""
        count = f" ({done}/{total} {unit})" if done is not None and total else ""
        print(f"{prefix} {data.get('message', '')}{count}", flush=True)
    elif kind == EventType.JOB_ESTIMATE.value:
        print(
            f"{prefix} estimated {data['total_s'] / 60:.1f} min for {data['frames']} frames "
            f"({data['basis']})",
            flush=True,
        )
    elif kind == EventType.STAGE_SKIPPED.value:
        print(f"{prefix} SKIPPED: {data.get('reason')}", flush=True)
    elif kind == EventType.WARNING.value:
        print(
            f"{prefix} {data.get('severity', '').upper()} {data.get('code')}: "
            f"{data.get('message')}",
            flush=True,
        )
    elif kind in (EventType.STAGE_STARTED.value, EventType.STAGE_COMPLETED.value):
        extra = f" in {data['duration_s']:.1f} s" if "duration_s" in data else ""
        print(f"{prefix} {kind.split('.')[1]}{extra}", flush=True)
    elif kind in (
        EventType.JOB_FAILED.value,
        EventType.JOB_COMPLETED.value,
        EventType.JOB_CANCELLED.value,
    ):
        print(f"{prefix} {kind.upper()} {data.get('reason', '')}", flush=True)


def run_local(
    video: Path,
    profile: ComputeProfile,
    out: Path,
    telemetry: Path | None = None,
    fps: float | None = None,
) -> int:
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("artifacts", "work", "preflight", "frames"):
        shutil.rmtree(out / sub, ignore_errors=True)
    print(f"NAZAR full pipeline · profile {profile.name} · {video}")
    print(f"Output: {out.resolve()}")
    gpu = GpuRuntime.from_env()
    runtime = Fast3rRuntime.from_env()
    print(f"  main GPU env : {gpu.main_python or 'NOT INSTALLED'}")
    print(f"  splat env    : {gpu.splat_python or 'NOT INSTALLED'}")
    print(f"  Fast3R       : {'ready' if isinstance(runtime, Fast3rRuntime) else runtime}")
    budget = profile.vram_budget_mib
    print(f"  GPU budget   : {f'{budget} MiB per model process' if budget else 'none (clg)'}")

    t0 = time.monotonic()
    print("Preflight (FFmpeg decode, quality, keyframes)...", flush=True)
    sha = _sha256(video)
    try:
        report = run_preflight(
            video,
            out / "preflight",
            asset_id="local",
            source_sha256=sha,
            max_duration_s=3600,
            telemetry_present=telemetry is not None,
        )
    except MediaError as exc:
        print(f"Video rejected: {exc.message}", file=sys.stderr)
        return 2
    print(f"  {report.analysis.frames_decoded} frames decoded ({time.monotonic() - t0:.1f} s)")
    where = ComputeTarget.COLLEGE if profile.strict else ComputeTarget.LOCAL
    cap = MAX_FRAMES[where]
    try:
        plan = sample_job_frames(
            video, report, fps, cap, out / "frames", auto_max=AUTO_MAX_FRAMES[where]
        )
    except MediaError as exc:
        print(f"Frames could not be sampled: {exc.message}", file=sys.stderr)
        return 2
    print(
        f"  {len(plan.keyframes)} frames "
        + ("(automatic) " if fps is None else f"at {fps:g} fps ")
        + f"(effective {plan.fps_effective:g} fps, cap {cap})"
        + (f"; lowered from {plan.wanted} by the cap" if plan.clamped else "")
    )

    track = None
    if telemetry is not None:
        try:
            track = track_to_json(parse_telemetry(telemetry.name, telemetry.read_bytes()))
        except TelemetryError as exc:
            print(f"Telemetry rejected: {exc}", file=sys.stderr)
            return 2

    client = LocalClient(out / "frames", out, track)
    lease = WorkerLease(
        lease_id="local",
        job_id=f"local-{uuid.uuid4().hex[:8]}",
        mission_id="local",
        pipeline=PipelineMode.STANDARD,
        attempt=1,
        lease_expires_at=datetime.now(UTC) + timedelta(days=1),
        video_asset_id="local",
        keyframe_count=len(plan.keyframes),
        video_sha256=sha,
        telemetry_asset_id="local" if track is not None else None,
    )
    adapters = build_standard_pipeline(runtime, profile=profile, gpu=gpu, work_dir=out / "work")
    forecast = Forecast.for_worker(profile, gpu)
    outcome = JobRun(client, lease, adapters, heartbeat_s=3600, forecast=forecast).execute()
    total_s = time.monotonic() - t0
    (out / "events.json").write_text(json.dumps(client.events_log, indent=1), encoding="utf-8")
    return _summary(out, client, profile, outcome, total_s)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(
    out: Path, client: LocalClient, profile: ComputeProfile, outcome: str, total_s: float
) -> int:
    print()
    print("=" * 78)
    print(f"RESULT: job {outcome.upper()} in {total_s / 60:.1f} min · profile {profile.name}")
    report_path = client.artifacts / "report.json"
    if not report_path.is_file():
        failed = [e for e in client.events_log if e["type"] == EventType.JOB_FAILED.value]
        if failed:
            print(f"Reason: {(failed[-1].get('data') or {}).get('reason')}")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contributions: dict[str, Any] = report.get("contributions", {})
    peaks: dict[str, int] = report["resources"].get("gpu_peak_vram_mib_by_stage", {})
    budget = report["resources"].get("gpu_budget_mib")
    print()
    print("Per-model contribution")
    problems = []
    for key in MODEL_STAGES:
        c = contributions.get(key)
        if c is None:
            print(f"  {key:22} -")
            continue
        if not c.get("applied"):
            print(f"  {key:22} NOT APPLIED: {c.get('reason')}")
            problems.append(f"{key}: {c.get('reason')}")
            continue
        peak = peaks.get(key)
        mem = f" · peak {peak} MiB" if peak is not None else ""
        print(f"  {key:22} {_one_line(key, c)}{mem}")
    sensor = contributions.get("sensor_optimization")
    if sensor is not None and not sensor.get("applied"):
        print(f"  {'sensor_optimization':22} not run: {sensor.get('reason')}")
    print()
    frame = report["coordinate_frame"]
    print(
        f"Scale: {frame['scale_status']}"
        + (f", 1 unit = {frame['metres_per_unit']:.4f} m" if frame.get("metres_per_unit") else "")
    )
    conf = report["confidence"]
    print(f"Cloud: {conf['observed']:,} observed / {conf['estimated']:,} estimated points")
    if "mesh_vertices" in conf:
        mv = conf["mesh_vertices"]
        print(
            f"Mesh vertices: {mv['observed']:,} observed / {mv['estimated']:,} estimated / "
            f"{mv['inferred']:,} inferred"
        )
    if budget:
        over = {k: v for k, v in peaks.items() if v > budget}
        verdict = "within" if not over else f"OVER on {over}"
        print(
            f"GPU memory: peak {max(peaks.values(), default=0)} MiB, budget {budget} MiB: {verdict}"
        )
        if over:
            problems.append(f"GPU budget exceeded: {over}")
    print()
    print("Artifacts:")
    for a in client.published:
        if a["key"].startswith("preview-"):
            continue
        print(f"  {a['kind']:20} {a['bytes'] / 2**20:8.2f} MiB  {a['file']}")
    print(f"Model inputs/outputs: {out / 'work'}")
    print(f"Full report: {report_path}")
    if outcome != "completed":
        return 1
    if problems:
        print()
        print("Completed, but not every model contributed:")
        for p in problems:
            print(f"  - {p}")
        return 3
    return 0


def _one_line(key: str, c: dict[str, Any]) -> str:
    if key == "dynamic_analysis":
        o = c["objects"]
        return (
            f"{o['moving']} moving / {o['static']} static objects, "
            f"{100 * c['masked_pixel_share']:.2f}% pixels masked"
        )
    if key == "geometry":
        return f"{c.get('method')}: {c.get('placed_keyframes')} views in {c.get('passes')} pass(es)"
    if key == "depth_priors":
        scale = c.get("scale", {})
        s = (
            f", implied 1 unit = {scale['metres_per_unit_estimated']:.2f} m (reported only)"
            if scale
            else ""
        )
        return f"{c.get('model')}: depth + normals{s}"
    if key == "dense_reconstruction":
        return (
            f"{c.get('final_points', 0):,} points, {c.get('from_depth_prior', 0):,} from the "
            f"depth prior, {c.get('points_with_prior_normal', 0):,} with prior normals"
        )
    if key == "gaussian_splat":
        return (
            f"{c.get('gaussians', 0):,} Gaussians, PSNR train {c.get('train_psnr_db')} / "
            f"held-out {c.get('test_psnr_db')} dB"
        )
    if key == "surface_processing":
        return (
            f"{c.get('vertices', 0):,} vertices, {c.get('triangles', 0):,} triangles, "
            f"{100 * c.get('inferred_vertex_share', 0):.1f}% inferred"
        )
    return "applied"


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "local", help="Run the full pipeline on a video file, no API (prints a summary)."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "--how",
        choices=sorted(PROFILES),
        required=True,
        help="local = 6 GB laptop GPU (memory-capped); clg = college GPU.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"Frames per second of video to reconstruct from ({FPS_MIN:g}-{FPS_MAX:g}). "
        "Default: automatic, about one frame per second but at least 24 frames.",
    )
    p.add_argument("--telemetry", type=Path, default=None, help="Optional DJI .SRT or .CSV.")
    p.add_argument(
        "--out", type=Path, default=None, help="Output folder (default var/runs/<video>-<how>)."
    )


def main(args: argparse.Namespace) -> int:
    out = args.out or Path("var") / "runs" / f"{args.video.stem}-{args.how}"
    if args.fps is not None and not FPS_MIN <= args.fps <= FPS_MAX:
        print(f"--fps must be {FPS_MIN:g}-{FPS_MAX:g}.", file=sys.stderr)
        return 2
    return run_local(
        args.video.resolve(), PROFILES[args.how], out.resolve(), args.telemetry, args.fps
    )
