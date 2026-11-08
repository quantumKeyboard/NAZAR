"""Standard pipeline: video keyframes -> explainable 3D model, one model per stage.

    preprocessing     keyframes chosen at upload
    dynamic_analysis  YOLO + SAM 2: masks of objects that actually moved
    geometry          Fast3R (masked): poses + dense pointmaps; CPU SfM baseline as fallback
    depth_priors      Metric3D-v2: metric depth + normals, compared with Fast3R per view
    sensor_opt.       Sim(3) GNSS alignment (telemetry only)
    dense             fusion of Fast3R + priors (Open3D-ready cloud), then a Gaussian splat
    surface           height-map solid with a texture baked from the full-resolution frames
                      (Open3D screened Poisson only as a fallback)
    analysis/export   confidence, per-model contributions, PLY/LAS/GLB/mesh/splat + report

Each GPU model runs in its own process with its own environment (`gpu.py`) and, on the
`local` profile, under a GPU-memory cap (`profile.py`). Every stage records what it
contributed in the report's `contributions`. On `local`, a step that cannot fit the memory
budget is skipped with a visible reason; on `clg` every model must run. When Fast3R fails on
`local`, the job falls back to the CPU baseline and says so in a warning, the report and every
artifact's provenance; it never substitutes silently.
"""

import hashlib
import io
import json
import platform
import shutil
import tempfile
import time
import weakref
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import scipy
from scipy.spatial import cKDTree

from nazar_contracts import (
    ArtifactCreate,
    ArtifactKind,
    Bounds3D,
    CoordinateFrame,
    FrameRange,
    FrameType,
    Provenance,
    ScaleStatus,
    StageName,
    WorkerLease,
)
from nazar_media.telemetry import track_from_json
from nazar_worker import fast3r, heightfield
from nazar_worker.fast3r import DenseCloud, Fast3rFailed, Fast3rRuntime, run_fast3r
from nazar_worker.fusion import SOURCE_FAST3R, Fused, Priors, fuse
from nazar_worker.georef import Alignment, align
from nazar_worker.glb import write_glb, write_mesh_glb
from nazar_worker.gpu import GpuRuntime, RunnerFailed, run_runner
from nazar_worker.las import write_las
from nazar_worker.ply import ESTIMATED, OBSERVED, write_ply_arrays
from nazar_worker.profile import LOCAL, ComputeProfile
from nazar_worker.sfm import SfmFailed, SfmSettings, Snapshot, reconstruct
from nazar_worker.stages import StageAdapter, StageContext, StageFailed

SFM = "sfm"
FAST3R = "fast3r"
PRODUCERS = {SFM: "nazar-worker/sfm-baseline", FAST3R: "nazar-worker/fast3r"}
MODELS = {
    SFM: "Classical SfM baseline (OpenCV SIFT + PnP + SciPy bundle adjustment)",
    FAST3R: fast3r.MODEL,
}
LOCAL_FRAME = CoordinateFrame(
    type=FrameType.LOCAL_RECONSTRUCTION, units="unitless", scale_status=ScaleStatus.UNKNOWN
)
FRAME_NOTE = "local frame, +Y up (estimated from dominant plane), unitless, unknown absolute scale"
PLY_NOTES = {
    SFM: f"NAZAR classical SfM baseline - sparse points from the video keyframes; {FRAME_NOTE}",
    FAST3R: f"NAZAR Fast3R - dense points predicted from the input video keyframes; {FRAME_NOTE}",
}
OBSERVED_MIN_VIEWS = 3
OBSERVED_MAX_PX = 1.0
MIN_FAST3R_VIEWS = 4

# The model stages: producer id and model name in artifact provenance.
STAGE_MODELS = {
    "dynamic_analysis": ("nazar-worker/yolo-sam2", "YOLO11 (Ultralytics) + SAM 2.1 (Meta)"),
    "depth_priors": ("nazar-worker/metric3d", "Metric3D-v2"),
    "dense_reconstruction": ("nazar-worker/fusion", "Fast3R + Metric3D-v2 fusion"),
    "gaussian_splat": ("nazar-worker/gsplat", "3D Gaussian splatting (gsplat), InstantSplat-style"),
    "surface_processing": (
        "nazar-worker/heightfield",
        "Height-map fusion + texture baking (Open3D Poisson as fallback)",
    ),
}


def _provenance(method: str) -> Provenance:
    if method == FAST3R:
        version = f"{fast3r.SOURCE_COMMIT}; weights {fast3r.WEIGHTS_REVISION}"
        settings = json.dumps(
            {
                "keep_min_conf": fast3r.KEEP_MIN_CONF,
                "keep_quantile": fast3r.KEEP_QUANTILE,
                "observed_min_conf": fast3r.OBSERVED_MIN_CONF,
            },
            sort_keys=True,
        ).encode()
    else:
        version = f"opencv {cv2.__version__}; scipy {scipy.__version__}"
        settings = json.dumps(SfmSettings().__dict__, sort_keys=True).encode()
    return Provenance(
        producer=PRODUCERS[method],
        model=MODELS[method],
        model_version=version,
        config_hash=hashlib.sha256(settings).hexdigest()[:16],
        test_pipeline=False,
    )


@dataclass
class _Run:
    """State handed from stage to stage within one job attempt."""

    runtime: Fast3rRuntime | str
    profile: ComputeProfile = LOCAL
    gpu: GpuRuntime = field(default_factory=GpuRuntime.from_env)
    work_dir: Path | None = None
    """Where model inputs/outputs live during the job (kept when given; else a temp dir)."""
    images: list[npt.NDArray[np.uint8]] = field(default_factory=list)
    keyframe_index: list[int] = field(default_factory=list)
    """Source video frame index of each keyframe image."""
    keyframe_times: list[float] = field(default_factory=list)
    """Video time (s) of each keyframe image."""
    telemetry: dict[str, Any] | None = None
    """What the sensor stage was given, for the report (None: no telemetry stage ran)."""
    alignment: Alignment | None = None
    method: str = SFM
    fallback_reason: str | None = None
    snapshot: Snapshot | None = None
    dense: DenseCloud | None = None
    gpu_stats: dict[str, Any] = field(default_factory=dict)
    """Fast3R runner statistics (device, load time, peak memory)."""
    keep: npt.NDArray[np.bool_] | None = None
    final: tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | None = (
        None
    )
    """The published final cloud: points, colours, confidence classes."""
    timings: dict[str, float] = field(default_factory=dict)
    first_preview_s: float | None = None
    started: float = field(default_factory=time.monotonic)
    masks_dir: Path | None = None
    """Moving-object masks (one PNG per keyframe), when dynamic analysis ran."""
    priors: Priors | None = None
    fused: Fused | None = None
    """The fused cloud after outlier filtering (same order as `final`)."""
    contributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    """What each stage/model added to the result, for the report."""
    gpu_peaks: dict[str, int] = field(default_factory=dict)
    _work: Path | None = None

    def work(self, sub: str = "") -> Path:
        if self._work is None:
            if self.work_dir is not None:
                self._work = self.work_dir.resolve()  # runners run from their own folder
                self._work.mkdir(parents=True, exist_ok=True)
            else:
                self._work = Path(tempfile.mkdtemp(prefix="nazar-job-"))
                weakref.finalize(self, shutil.rmtree, str(self._work), True)
        path = self._work / sub if sub else self._work
        path.mkdir(parents=True, exist_ok=True)
        return path

    def frame(self) -> CoordinateFrame:
        """Coordinates stay local; an accepted sensor alignment adds a metric scale."""
        if self.alignment is not None and self.alignment.scale_accepted:
            return CoordinateFrame(
                type=FrameType.LOCAL_RECONSTRUCTION,
                units="unitless",
                scale_status=ScaleStatus.METRIC_SENSOR,
                metres_per_unit=self.alignment.metres_per_unit,
            )
        return LOCAL_FRAME

    def centres(self) -> npt.NDArray[np.float64]:
        if self.dense is not None:
            return self.dense.centers
        assert self.snapshot is not None
        return self.snapshot.centers

    # The final geometry, whichever method produced it.
    def points(self) -> npt.NDArray[np.float64]:
        if self.dense is not None:
            return self.dense.points
        assert self.snapshot is not None
        return self.snapshot.points

    def placed(self) -> list[int]:
        if self.dense is not None:
            return self.dense.views
        assert self.snapshot is not None
        return self.snapshot.registered

    def observed_mask(self) -> npt.NDArray[np.bool_]:
        assert self.keep is not None
        if self.final is not None:
            return np.asarray(self.final[2] == OBSERVED)
        if self.dense is not None:
            return np.asarray(self.dense.conf[self.keep] >= fast3r.OBSERVED_MIN_CONF)
        assert self.snapshot is not None
        views, errs = self.snapshot.views[self.keep], self.snapshot.errors[self.keep]
        return np.asarray((views >= OBSERVED_MIN_VIEWS) & (errs <= OBSERVED_MAX_PX))


def _declare(
    run: _Run,
    key: str,
    kind: ArtifactKind,
    media_type: str,
    content: bytes,
    *,
    xyz: npt.NDArray[np.float64] | None = None,
    frames: FrameRange | None = None,
    lod: int | None = None,
    final: bool = False,
    spatial: bool = False,
    stage: str | None = None,
) -> ArtifactCreate:
    """`stage` names a model stage (see STAGE_MODELS) whose model produced the artifact;
    otherwise the geometry method is the producer."""
    box = None
    if xyz is not None and len(xyz):
        box = Bounds3D(min=xyz.min(axis=0).tolist(), max=xyz.max(axis=0).tolist())
    return ArtifactCreate(
        key=key,
        kind=kind,
        media_type=media_type,
        byte_size=len(content),
        checksum_sha256=hashlib.sha256(content).hexdigest(),
        frame=run.frame() if spatial or xyz is not None else None,
        bounds=box,
        point_count=len(xyz) if xyz is not None else None,
        lod=lod,
        source_frames=frames,
        provenance=_provenance(run.method) if stage is None else _stage_provenance(stage, run),
        final=final,
    )


def _stage_provenance(stage: str, run: "_Run") -> Provenance:
    producer, model = STAGE_MODELS[stage]
    return Provenance(
        producer=producer,
        model=model,
        model_version=f"profile {run.profile.name}",
        config_hash=hashlib.sha256(
            json.dumps(run.profile.report(), sort_keys=True).encode()
        ).hexdigest()[:16],
        test_pipeline=False,
    )


def _zip(folder: Path, pattern: str = "*") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob(pattern)):
            if path.is_file():
                zf.write(path, path.relative_to(folder).as_posix())
    return buf.getvalue()


def _frames(run: _Run, used: list[int] | None = None) -> FrameRange | None:
    idx = [run.keyframe_index[i] for i in (used if used is not None else range(len(run.images)))]
    if not idx:
        return None
    return FrameRange(first_index=min(idx), last_index=max(idx), count=len(idx))


class _Stage:
    def __init__(self, run: _Run) -> None:
        self.run_state = run

    def skip_reason(self, lease: WorkerLease) -> str | None:
        return None

    def _missing_or_skip(self, missing: str | None) -> str | None:
        """A model that is not installed: skipped with the reason on `local`; on `clg` the
        stage starts and fails, because that profile must run every model."""
        return missing if missing and not self.run_state.profile.strict else None

    def _skipped(self, key: str, reason: str | None) -> str | None:
        """Record a skip (with its reason) for the report, and return the reason."""
        if reason is not None:
            self.run_state.contributions[key] = {"applied": False, "reason": reason}
        return reason

    def _require(self, missing: str | None, what: str) -> None:
        if missing:
            raise StageFailed(
                f"{what}: {missing} The {self.run_state.profile.name} profile runs every model."
            )

    def _gpu_step(
        self,
        ctx: StageContext,
        key: str,
        what: str,
        script: str,
        args: list[str],
        steps: dict[str, tuple[str, str]],
        *,
        splat: bool = False,
    ) -> dict[str, Any] | None:
        """Run a GPU runner. Returns its result, or None when the step did not fit the local
        GPU-memory budget (then it is skipped visibly). Any other failure fails the job."""
        run = self.run_state
        python = run.gpu.splat_python if splat else run.gpu.main_python
        assert python is not None

        def on_event(event: dict[str, Any]) -> None:
            kind = event.get("event")
            if kind == "progress" and event.get("step") in steps:
                message, unit = steps[str(event["step"])]
                ctx.progress(message, event.get("done"), event.get("total"), unit)
            elif kind == "loaded":
                ctx.progress(f"{what}: model loaded ({event.get('model', 'ok')})", None, None, None)
            elif kind == "retry":
                ctx.warn(
                    f"{key}_retry",
                    f"{what}: {event.get('reason', 'retrying')}; continuing with "
                    f"{event.get('gaussians') or event.get('model') or 'smaller settings'}.",
                    "info",
                )

        try:
            done = run_runner(python, script, args, ctx.check_cancel, on_event)
        except RunnerFailed as exc:
            budget = run.profile.vram_budget_mib
            if exc.oom and not run.profile.strict:
                reason = (
                    f"{what} needs more GPU memory than the local budget ({budget} MiB), so it "
                    f"was skipped on the local GPU; the college profile runs it. ({exc})"
                )
                ctx.warn(f"{key}_skipped_gpu_memory", reason[:480], "warning")
                run.contributions[key] = {"applied": False, "reason": reason}
                return None
            where = "on the college GPU" if run.profile.strict else "on the local GPU"
            raise StageFailed(f"{what} failed {where}: {exc}", retryable=True) from exc
        if done.get("peak_vram_mib") is not None:
            run.gpu_peaks[key] = int(done["peak_vram_mib"])
            ctx.gpu(int(done["peak_vram_mib"]), int(done.get("budget_mib") or 0))
        return done


class Preprocessing(_Stage):
    """Fetches the keyframes chosen at upload (quality + viewpoint novelty) and decodes them."""

    def run(self, ctx: StageContext) -> None:
        t0 = time.monotonic()
        report = ctx.preflight()
        keyframes = report.get("keyframes", [])
        if len(keyframes) < 3:
            raise StageFailed(
                f"The preflight selected only {len(keyframes)} keyframes; at least 3 are needed."
            )
        for n, kf in enumerate(keyframes, start=1):
            ctx.check_cancel()
            data = np.frombuffer(ctx.keyframe(int(kf["image"])), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise StageFailed(f"Keyframe {kf['image']} could not be decoded.")
            self.run_state.images.append(img.astype(np.uint8))
            cv2.imwrite(
                str(self.run_state.work("frames") / f"{n - 1:04d}.jpg"),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            self.run_state.keyframe_index.append(int(kf["index"]))
            self.run_state.keyframe_times.append(float(kf["t"]))
            if n % 8 == 0 or n == len(keyframes):
                ctx.progress("Loaded keyframes from preflight", n, len(keyframes), "keyframes")
        self.run_state.timings["preprocessing_s"] = round(time.monotonic() - t0, 3)


class DynamicAnalysis(_Stage):
    """YOLO candidates -> motion check against neighbouring keyframes -> SAM 2 masks. The masks
    run *before* geometry, so moving objects affect neither Fast3R's poses nor the cloud."""

    def _missing(self) -> str | None:
        p, g = self.run_state.profile, self.run_state.gpu
        return g.missing(f"yolo/{p.yolo_weights}", f"sam2/{p.sam2_checkpoint}")

    def skip_reason(self, lease: WorkerLease) -> str | None:
        return self._skipped("dynamic_analysis", self._missing_or_skip(self._missing()))

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        self._require(self._missing(), "Moving-object masking cannot run")
        t0 = time.monotonic()
        out = run.work("dynamic")
        n = len(run.images)
        ctx.progress("YOLO: looking for objects that can move", 0, n, "keyframes")
        done = self._gpu_step(
            ctx,
            "dynamic_analysis",
            "YOLO + SAM 2 masking",
            "dynamic_runner.py",
            [
                "--frames",
                str(run.work("frames")),
                "--out",
                str(out),
                "--yolo",
                str(run.gpu.weights / "yolo" / run.profile.yolo_weights),
                "--sam2-ckpt",
                str(run.gpu.weights / "sam2" / run.profile.sam2_checkpoint),
                "--sam2-cfg",
                run.profile.sam2_config,
                "--imgsz",
                str(run.profile.yolo_imgsz),
                "--budget-mib",
                str(run.profile.vram_budget_mib),
            ],
            {
                "detect": ("YOLO: detecting vehicles, people and boats", "keyframes"),
                "segment": ("SAM 2: segmenting the objects that moved", "keyframes"),
            },
        )
        if done is None:
            return
        run.masks_dir = out / "masks"
        objects = done["objects"]
        ctx.progress(
            f"{objects['moving']} moving, {objects['static']} static and {objects['unknown']} "
            f"undecided objects; {100 * done['masked_pixel_share']:.2f}% of pixels masked",
            n,
            n,
            "keyframes",
        )
        archive = _zip(out)
        ctx.publish(
            _declare(
                run,
                "dynamic-masks",
                ArtifactKind.DYNAMIC_MASKS,
                "application/zip",
                archive,
                frames=_frames(run),
                final=True,
                stage="dynamic_analysis",
            ),
            archive,
        )
        run.contributions["dynamic_analysis"] = {
            "applied": True,
            "what": "Masks of objects that moved between keyframes; they are excluded from "
            "Fast3R's poses, the point cloud, the mesh and the splat's training loss.",
            "detections": done["detections"],
            "objects": objects,
            "frames_with_moving_objects": done["frames_with_moving_objects"],
            "masked_pixel_share": done["masked_pixel_share"],
            "rule": "moving = the object left its ground spot in a neighbouring keyframe "
            "(ground-plane homography; no same-class detection overlaps there and the image "
            "patch changed); parked objects are kept; no usable neighbour = undecided, kept.",
            "peak_vram_mib": done.get("peak_vram_mib"),
        }
        run.timings["dynamic_analysis_s"] = round(time.monotonic() - t0, 3)


class Geometry(_Stage):
    """Fast3R when available, else the SfM baseline; every intermediate state is published as
    a preview revision."""

    revision = 0

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        t0 = time.monotonic()
        if not isinstance(run.runtime, Fast3rRuntime) and run.profile.strict:
            raise StageFailed(
                f"Fast3R cannot run: {run.runtime} The {run.profile.name} profile runs every model."
            )
        if isinstance(run.runtime, Fast3rRuntime):
            try:
                self._fast3r(ctx, run.runtime)
            except Fast3rFailed as exc:
                if run.profile.strict:
                    raise StageFailed(
                        f"Fast3R failed on the college GPU: {exc}", retryable=True
                    ) from exc
                run.fallback_reason = str(exc)
                run.dense, run.gpu_stats = None, {}
                ctx.warn(
                    "fast3r_fallback",
                    "Fast3R could not run on this job, so the CPU baseline (classical "
                    f"structure-from-motion) was used instead. Reason: {str(exc)[:200]}",
                    "warning",
                )
        if run.dense is None:
            run.method = SFM
            self._sfm(ctx)
            run.contributions["geometry"] = {
                "applied": True,
                "method": MODELS[SFM],
                "what": "Sparse camera poses and feature points (CPU fallback).",
                "fallback_reason": run.fallback_reason,
            }
        else:
            stats = run.dense.stats
            run.contributions["geometry"] = {
                "applied": True,
                "method": MODELS[FAST3R],
                "what": "Camera poses and dense per-pixel pointmaps: the base geometry.",
                "placed_keyframes": stats.get("placed_keyframes"),
                "pixels_kept": stats.get("pixels_kept"),
                "passes": stats.get("chunks"),
                "views_per_pass": stats.get("max_views"),
                "moving_object_masks_used": run.masks_dir is not None,
                "peak_vram_mib": stats.get("peak_vram_mib"),
            }
            if stats.get("peak_vram_mib") is not None:
                run.gpu_peaks["geometry"] = int(stats["peak_vram_mib"])
                ctx.gpu(int(stats["peak_vram_mib"]), int(stats.get("budget_mib") or 0))
        run.timings["geometry_s"] = round(time.monotonic() - t0, 3)

    def _fast3r(self, ctx: StageContext, runtime: Fast3rRuntime) -> None:
        run = self.run_state
        run.method = FAST3R
        total = len(run.images)
        max_views = runtime.max_views or run.profile.fast3r_max_views
        while True:
            per_pass = f"{max_views} views per pass" if max_views else "sizing passes to GPU memory"
            ctx.progress(f"Fast3R: loading the model ({per_pass})", 0, total, "keyframes")

            def on_event(event: dict[str, Any]) -> None:
                if event.get("event") == "loaded":
                    run.gpu_stats.update(device=event.get("device"), load_s=event.get("seconds"))
                    ctx.progress("Fast3R: predicting pointmaps", 0, total, "keyframes")
                elif event.get("event") == "retry":
                    ctx.warn(
                        "fast3r_smaller_batches",
                        f"The GPU ran out of memory; Fast3R continued with {event.get('max_views')}"
                        " views per pass (chunks are aligned on shared keyframes).",
                        "info",
                    )

            try:
                for cloud in run_fast3r(
                    runtime,
                    run.images,
                    run.work("fast3r"),
                    ctx.check_cancel,
                    on_event,
                    max_views=max_views,
                    masks=run.masks_dir,
                    budget_mib=run.profile.vram_budget_mib,
                ):
                    if cloud.final:
                        run.dense = cloud
                    else:
                        self._preview(ctx, cloud.points, cloud.colors, cloud.views, total)
                break
            except Fast3rFailed as exc:
                ran = exc.max_views or max_views
                if not exc.oom or ran // 2 < MIN_FAST3R_VIEWS:
                    raise
                max_views = ran // 2
                ctx.warn(
                    "fast3r_smaller_batches",
                    f"The GPU ran out of memory; Fast3R was rerun with {max_views} views per "
                    "pass (chunks are aligned on shared keyframes).",
                    "info",
                )
        assert run.dense is not None
        run.gpu_stats.update(run.dense.stats)
        self._preview(ctx, run.dense.points, run.dense.colors, run.dense.views, total)

    def _sfm(self, ctx: StageContext) -> None:
        run = self.run_state
        total = len(run.images)
        ctx.progress("Matching features between keyframes", 0, total, "keyframes")
        try:
            for snap in reconstruct(run.images, check_cancel=ctx.check_cancel):
                run.snapshot = snap
                if snap.final:
                    break
                self._preview(ctx, snap.points, snap.colors, snap.registered, total)
        except SfmFailed as exc:
            raise StageFailed(str(exc)) from exc
        final = run.snapshot
        assert final is not None and final.final
        snap = final
        missing = total - len(snap.registered)
        if missing:
            ctx.warn(
                "unregistered_keyframes",
                f"{missing} of {total} keyframes could not be placed (too little overlap or "
                "parallax with the rest); the cloud covers only what the others see.",
                "warning",
            )
        if len(snap.registered) < max(3, total // 3):
            raise StageFailed(
                f"Only {len(snap.registered)} of {total} keyframes could be reconstructed, "
                "which is too few for a trustworthy result."
            )

    def _preview(
        self,
        ctx: StageContext,
        points: npt.NDArray[np.float64],
        colors: npt.NDArray[np.uint8],
        placed: list[int],
        total: int,
    ) -> None:
        run = self.run_state
        self.revision += 1
        ply = write_ply_arrays(points, colors, [PLY_NOTES[run.method]])
        ctx.publish(
            _declare(
                run,
                f"preview-{self.revision}",
                ArtifactKind.POINT_CLOUD_PREVIEW,
                "application/vnd.nazar.ply",
                ply,
                xyz=points,
                frames=_frames(run, placed),
                lod=1,
            ),
            ply,
        )
        if run.first_preview_s is None:
            run.first_preview_s = round(time.monotonic() - run.started, 3)
        label = "Fast3R placed" if run.method == FAST3R else "Registered"
        ctx.progress(
            f"{label} {len(placed)} of {total} keyframes · {len(points):,} points",
            len(placed),
            total,
            "keyframes",
        )


class DepthPriors(_Stage):
    """Metric3D-v2 metric depth + surface normals per keyframe. They are compared with Fast3R
    view by view during fusion (scale evidence), give the mesher its normals, and fill holes
    where Fast3R had no confident pixels."""

    def _missing(self) -> str | None:
        run = self.run_state
        names = {
            "vit_small": "metric_depth_vit_small_800k.pth",
            "vit_large": "metric_depth_vit_large_800k.pth",
            "vit_giant2": "metric_depth_vit_giant2_800k.pth",
        }
        have = [
            m
            for m in run.profile.metric3d_models
            if (run.gpu.weights / "metric3d" / names[m]).is_file()
        ]
        if not (run.gpu.root / "metric3d-src" / "hubconf.py").is_file():
            return "The Metric3D source checkout is not installed on this worker."
        if not have:
            wanted = ", ".join(run.profile.metric3d_models)
            return f"No Metric3D-v2 weights for {wanted} are installed."
        return run.gpu.missing()

    def skip_reason(self, lease: WorkerLease) -> str | None:
        if self.run_state.dense is None:
            return self._skipped(
                "depth_priors",
                "Depth priors are compared with Fast3R's cameras and pointmaps; the CPU baseline "
                "ran instead, so there is nothing to compare them with.",
            )
        return self._skipped("depth_priors", self._missing_or_skip(self._missing()))

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        self._require(self._missing(), "Metric3D-v2 cannot run")
        dense = run.dense
        assert dense is not None and dense.npz is not None
        t0 = time.monotonic()
        data = np.load(dense.npz)
        width_in, height_in = (float(a) for a in data["input_size"])
        intr: dict[str, dict[str, float]] = {}
        for v, kf in enumerate(int(k) for k in data["views"]):
            factor, left, top = (float(a) for a in data["crop"][v])
            intr[f"{kf:04d}.jpg"] = {
                "fx": float(data["focal"][v]) / factor,
                "cx": (width_in / 2 + left) / factor,
                "cy": (height_in / 2 + top) / factor,
            }
        frames = run.work("depth-frames")
        for name in intr:
            shutil.copy2(run.work("frames") / name, frames / name)
        (run.work() / "intrinsics.json").write_text(json.dumps(intr), encoding="utf-8")
        out = run.work("depth")
        n = len(intr)
        ctx.progress("Metric3D-v2: loading the depth model", 0, n, "keyframes")
        done = self._gpu_step(
            ctx,
            "depth_priors",
            "Metric3D-v2 depth priors",
            "metric3d_runner.py",
            [
                "--frames",
                str(frames),
                "--intrinsics",
                str(run.work() / "intrinsics.json"),
                "--src",
                str(run.gpu.root / "metric3d-src"),
                "--weights",
                str(run.gpu.weights / "metric3d"),
                "--out",
                str(out),
                "--models",
                ",".join(run.profile.metric3d_models),
                "--budget-mib",
                str(run.profile.vram_budget_mib),
            ],
            {"infer": ("Metric3D-v2: metric depth and normals", "keyframes")},
        )
        if done is None:
            return
        depth: dict[int, npt.NDArray[np.float32]] = {}
        normals: dict[int, npt.NDArray[np.float32]] = {}
        for name in intr:
            arr = np.load(out / name.replace(".jpg", ".npz"))
            kf = int(name[:4])
            depth[kf] = arr["depth"].astype(np.float32)
            normals[kf] = arr["normals"].astype(np.float32)
        run.priors = Priors(depth, normals, str(done["model"]), done)
        archive = _zip(out, "*.jpg")
        ctx.publish(
            _declare(
                run,
                "depth-previews",
                ArtifactKind.DEPTH_MAPS,
                "application/zip",
                archive,
                frames=_frames(run, list(depth)),
                final=True,
                stage="depth_priors",
            ),
            archive,
        )
        run.contributions["depth_priors"] = {
            "applied": True,
            "model": done["model"],
            "variants_tried": done.get("tried"),
            "what": "Per-pixel metric depth and surface normals (camera frame) per keyframe; "
            "fusion compares them with Fast3R and the mesher uses the normals.",
            "normal_convention_detected": done.get("normal_convention"),
            "normals_facing_camera": done.get("normals_facing_camera"),
            "peak_vram_mib": done.get("peak_vram_mib"),
            "note": "Metric only as far as the intrinsics are right; NAZAR's are Fast3R's "
            "estimate, so anything derived is labelled estimated.",
        }
        ctx.progress(
            f"Metric3D-v2 ({done['model']}): depth and normals for {n} keyframes", n, n, "keyframes"
        )
        run.timings["depth_priors_s"] = round(time.monotonic() - t0, 3)


class SensorOptimization(_Stage):
    """Similarity alignment of the camera path to GNSS telemetry (see georef.py). Accepted
    claims add a metric scale (and an ENU georeference for exports); rejected ones keep the
    local frame, with the reason in the report. GTSAM pose-graph fusion is not used: one
    Sim(3) is the honest model for a single video with consumer GNSS."""

    def skip_reason(self, lease: WorkerLease) -> str | None:
        if lease.telemetry_asset_id:
            return None
        return self._skipped("sensor_optimization", "No telemetry was supplied with this mission.")

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        t0 = time.monotonic()
        doc = ctx.telemetry()
        if doc is None:
            run.telemetry = {
                "applied": False,
                "reason": "Telemetry was removed after the job "
                "was created; the reconstruction stays local.",
            }
            ctx.warn("telemetry_removed", run.telemetry["reason"], "warning")
            return
        track = track_from_json(doc)
        placed = run.placed()
        times = np.array([run.keyframe_times[i] for i in placed])
        ctx.progress(
            f"Aligning {len(placed)} camera positions to {len(track)} GNSS samples",
            None,
            None,
            None,
        )
        result = align(run.centres(), times, track)
        run.alignment = result
        run.telemetry = {
            "applied": result.scale_accepted,
            "format": track.format,
            "samples": len(track),
            "altitude_kind": track.altitude_kind.value,
            "dropped_samples": track.dropped,
            "gaps": len(track.gaps),
            "enu_origin": {
                "latitude": round(float(track.lat[0]), 7),
                "longitude": round(float(track.lon[0]), 7),
                "height_m": None if np.isnan(track.alt[0]) else round(float(track.alt[0]), 2),
                "height_kind": track.altitude_kind.value,
                "note": "First valid telemetry fix; ENU = east, north, up metres from it.",
            },
            **result.report(),
        }
        if result.scale_accepted:
            ctx.progress(
                f"Scale from GNSS: 1 unit = {result.metres_per_unit:.3f} m "
                f"(residual {result.residual_rms_m:.1f} m RMS over {result.used} keyframes)",
                result.used,
                len(placed),
                "keyframes",
            )
        else:
            ctx.progress(
                "Telemetry not applied: " + " ".join(result.reasons), 0, len(placed), "keyframes"
            )
        run.timings["sensor_optimization_s"] = round(time.monotonic() - t0, 3)


class Densify(_Stage):
    """The dense scene. With Fast3R: the pointmaps are fused with the Metric3D-v2 priors
    (normals, scale evidence, hole fill; moving objects excluded), outlier-filtered and
    published with the camera path, then a Gaussian splat is trained on that cloud and the
    keyframes. With the CPU baseline: the sparse SfM cloud, filtered."""

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        t0 = time.monotonic()
        source: npt.NDArray[np.uint8] | None = None
        if run.dense is not None:
            dense = run.dense
            assert dense.npz is not None and dense.transform is not None
            with_priors = run.priors is not None
            ctx.progress(
                "Fusing Fast3R pointmaps"
                + (" with Metric3D-v2 depth and normals" if with_priors else ""),
                None,
                None,
                None,
            )
            fused = fuse(
                dense.npz, dense.transform, dense.keep_thr, run.priors, run.profile.max_fused_points
            )
            keep = _statistical_inliers(fused.points)
            run.fused = Fused(
                fused.points[keep],
                fused.colors[keep],
                fused.conf[keep],
                fused.normals[keep],
                fused.source[keep],
                fused.cameras[keep],
                fused.stats,
            )
            pts, cols, source = run.fused.points, run.fused.colors, run.fused.source
            observed = (source == SOURCE_FAST3R) & (run.fused.conf >= fast3r.OBSERVED_MIN_CONF)
            labels = np.where(observed, OBSERVED, ESTIMATED).astype(np.uint8)
            centers, focal = dense.centers, dense.focal_px
            self._scale_from_priors(fused.stats)
            run.contributions["dense_reconstruction"] = {
                "applied": True,
                "what": "One cloud from all views: Fast3R points (moving objects removed), "
                "Metric3D normals attached, low-confidence holes filled from the scaled prior.",
                **fused.stats,
                "outliers_removed": int((~keep).sum()),
                "final_points": int(keep.sum()),
                "from_depth_prior": int((source != SOURCE_FAST3R).sum()),
                "points_with_prior_normal": int(
                    (np.linalg.norm(run.fused.normals, axis=1) > 0.5).sum()
                ),
            }
        else:
            assert run.snapshot is not None
            pts, cols = run.snapshot.points, run.snapshot.colors
            centers, focal = run.snapshot.centers, run.snapshot.focal_px
            keep = _statistical_inliers(pts)
            pts, cols = pts[keep], cols[keep]
            run.keep = keep
            labels = np.where(run.observed_mask(), OBSERVED, ESTIMATED).astype(np.uint8)
        run.keep = keep
        what = "fused dense cloud" if run.dense is not None else "sparse cloud"
        ctx.progress(
            f"Filtered {what} ({int((~keep).sum()):,} outliers removed)",
            int(keep.sum()),
            len(keep),
            "points",
        )
        run.final = (pts, cols, labels)
        ply = write_ply_arrays(pts, cols, [PLY_NOTES[run.method]], labels, source)
        ctx.publish(
            _declare(
                run,
                "point-cloud",
                ArtifactKind.POINT_CLOUD,
                "application/vnd.nazar.ply",
                ply,
                xyz=pts,
                frames=_frames(run, run.placed()),
                lod=0,
                final=True,
            ),
            ply,
        )
        poses = [
            {
                "keyframe": i,
                "frame_index": run.keyframe_index[i],
                "time_s": round(run.keyframe_times[i], 4),
                "position": c.round(5).tolist(),
            }
            for i, c in zip(run.placed(), centers, strict=True)
        ]
        cam_source = (
            "Camera poses estimated by Fast3R (PnP on its pointmaps)"
            if run.method == FAST3R
            else "Camera centres from the classical SfM baseline"
        )
        traj = json.dumps(
            {
                "frame": "local_reconstruction",
                "units": "unitless",
                "metres_per_unit": run.frame().metres_per_unit,
                "note": f"{cam_source}, same frame as the cloud.",
                "focal_px_estimated": round(focal, 2),
                "poses": poses,
            },
            indent=1,
        ).encode()
        ctx.publish(
            _declare(
                run,
                "camera-trajectory",
                ArtifactKind.CAMERA_TRAJECTORY,
                "application/json",
                traj,
                frames=_frames(run, run.placed()),
                final=True,
                spatial=True,
            ),
            traj,
        )
        run.timings["dense_reconstruction_s"] = round(time.monotonic() - t0, 3)
        if run.dense is not None:
            t1 = time.monotonic()
            self._splat(ctx)
            run.timings["gaussian_splat_s"] = round(time.monotonic() - t1, 3)

    def _scale_from_priors(self, stats: dict[str, Any]) -> None:
        """Record the scale Metric3D implies. It is reported, never applied: it inherits the
        error of Fast3R's estimated focal length, which per-view agreement cannot reveal (on
        sample1 it came out about 2.2x smaller than the Colosseum's known size)."""
        run = self.run_state
        mpu = stats.get("metres_per_unit_estimated")
        if mpu is None or run.priors is None:
            return
        detail: dict[str, Any] = {
            "metres_per_unit_estimated": mpu,
            "scale_spread_rel": stats.get("scale_spread_rel"),
            "views_agreeing": stats.get("prior_views_agreeing"),
            "median_rel_err_vs_fast3r": stats.get("prior_median_rel_err"),
            "applied_to_outputs": False,
            "why_not_applied": "Metric depth scales with the focal length, and NAZAR's is Fast3R's "
            "estimate; agreement between views shows consistency, not absolute accuracy. Use "
            "GNSS telemetry or a known length for metric output.",
        }
        if run.alignment is not None and run.alignment.scale_accepted:
            assert run.alignment.metres_per_unit is not None
            gnss = run.alignment.metres_per_unit
            detail["vs_gnss_scale_rel_diff"] = round(abs(mpu - gnss) / gnss, 4)
        run.contributions.setdefault("depth_priors", {})["scale"] = detail

    def _splat(self, ctx: StageContext) -> None:
        run = self.run_state
        dense, fused = run.dense, run.fused
        assert dense is not None and dense.npz is not None and dense.transform is not None
        assert fused is not None
        missing = run.gpu.missing(splat=True)
        if missing:
            self._require(missing, "The Gaussian splat cannot be trained")
            run.contributions["gaussian_splat"] = {"applied": False, "reason": missing}
            ctx.warn("gaussian_splat_unavailable", missing, "info")
            return
        views, cams = _camera_records(run)
        work = run.work("splat")
        (work / "cameras.json").write_text(json.dumps({"cameras": cams}), encoding="utf-8")
        np.savez(work / "init.npz", points=fused.points.astype(np.float32), colors=fused.colors)
        p = run.profile
        done = self._gpu_step(
            ctx,
            "gaussian_splat",
            "Gaussian splat training",
            "splat_runner.py",
            [
                "--frames",
                str(run.work("frames")),
                "--cameras",
                str(work / "cameras.json"),
                "--points",
                str(work / "init.npz"),
                "--out",
                str(work),
                *(["--masks", str(run.masks_dir)] if run.masks_dir is not None else []),
                "--iterations",
                str(p.splat_iterations),
                "--max-gaussians",
                str(p.splat_max_gaussians),
                "--sh-degree",
                str(p.splat_sh_degree),
                "--budget-mib",
                str(p.vram_budget_mib),
            ],
            {"train": ("Gaussian splat: optimising Gaussians and camera poses", "iterations")},
            splat=True,
        )
        if done is None:
            return
        scene = (work / "scene.ply").read_bytes()
        ctx.publish(
            _declare(
                run,
                "gaussian-scene",
                ArtifactKind.GAUSSIAN_SCENE,
                "application/vnd.nazar.gaussians+ply",
                scene,
                xyz=fused.points,
                frames=_frames(run, views),
                final=True,
                spatial=True,
                stage="gaussian_splat",
            ),
            scene,
        )
        renders_dir = work / "renders"
        if renders_dir.is_dir() and any(renders_dir.iterdir()):
            renders = _zip(renders_dir)
            ctx.publish(
                _declare(
                    run,
                    "splat-renders",
                    ArtifactKind.EXPORT,
                    "application/zip",
                    renders,
                    final=True,
                    stage="gaussian_splat",
                ),
                renders,
            )
        keys = (
            "gaussians",
            "iterations",
            "sh_degree",
            "train_views",
            "test_views",
            "train_psnr_db",
            "test_psnr_db",
            "pose_refinement_mean_shift",
            "oom_retries",
            "peak_vram_mib",
        )
        run.contributions["gaussian_splat"] = {
            "applied": True,
            "what": "A photoreal view-dependent scene: Gaussians started on the fused cloud and "
            "optimised with the camera poses against the keyframes (moving objects masked out "
            "of the loss). For viewing, not for measuring.",
            **{k: done.get(k) for k in keys},
            "test_protocol": "every 8th keyframe held out; scored after a pose-only alignment "
            "with the Gaussians frozen (InstantSplat protocol)",
        }
        ctx.progress(
            f"Gaussian splat: {done['gaussians']:,} Gaussians, held-out PSNR "
            f"{done.get('test_psnr_db')} dB",
            p.splat_iterations,
            p.splat_iterations,
            "iterations",
        )


def _camera_records(run: _Run) -> tuple[list[int], list[dict[str, Any]]]:
    """Fast3R's camera for every placed keyframe, in the NAZAR world frame (OpenCV camera axes:
    x right, y down, z forward), with intrinsics in pixels of the full-resolution keyframe.
    Used by the Gaussian splat and by the surface texture."""
    dense = run.dense
    assert dense is not None and dense.npz is not None and dense.transform is not None
    tf = dense.transform
    data = np.load(dense.npz)
    width_in, height_in = (float(a) for a in data["input_size"])
    views = [int(k) for k in data["views"]]
    cams: list[dict[str, Any]] = []
    for v, kf in enumerate(views):
        factor, left, top = (float(a) for a in data["crop"][v])
        c2w = np.eye(4)
        c2w[:3, :3] = tf.rotation @ data["c2w"][v][:3, :3]
        c2w[:3, 3] = tf.points(data["c2w"][v][:3, 3][None])[0]
        h, w = run.images[kf].shape[:2]
        f = float(data["focal"][v]) / factor
        cams.append(
            {
                "image": f"{kf:04d}.jpg",
                "c2w": c2w.round(8).tolist(),
                "fx": f,
                "fy": f,
                "cx": (width_in / 2 + left) / factor,
                "cy": (height_in / 2 + top) / factor,
                "width": w,
                "height": h,
                "split": "test" if len(views) >= 8 and v % 8 == 4 else "train",
            }
        )
    return views, cams


class Surface(_Stage):
    """The solid, textured model.

    A single forward pass sees the ground and the tops of things from one side, so the scene is
    a height map. The fused points are binned into a grid whose cells keep a robust top height
    (outliers and noise are removed, small holes are bridged and labelled inferred), the grid is
    closed into a solid block (walls and a base), and a top-down texture is baked from the
    full-resolution video frames: every texel is projected into the frames that see it best.
    That runs on the CPU in this process. Only when the scene cannot be a height map (too
    sparse, or mostly unobserved) does the stage fall back to Open3D screened Poisson in the
    GPU environment, which gives per-vertex colours but no texture.
    """

    def _missing(self) -> str | None:
        return self.run_state.gpu.missing()

    def skip_reason(self, lease: WorkerLease) -> str | None:
        if self.run_state.fused is None:
            return self._skipped(
                "surface_processing",
                "Skipped: a sparse cloud cannot support a meaningful surface; it would be mostly "
                "invented.",
            )
        return None

    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        assert run.fused is not None and run.final is not None
        t0 = time.monotonic()
        try:
            self._height_map(ctx)
        except heightfield.HeightfieldError as exc:
            reason = str(exc)
            ctx.warn(
                "surface_heightfield_unavailable",
                f"A height-map surface was not possible ({reason}); trying screened Poisson.",
                "warning",
            )
            missing = self._missing()
            if missing:
                if run.profile.strict:
                    raise StageFailed(
                        f"No surface could be built: the height map failed ({reason}) and the "
                        f"Poisson fallback cannot run: {missing}"
                    ) from exc
                run.contributions["surface_processing"] = {
                    "applied": False,
                    "reason": f"The height-map surface failed ({reason}) and the Poisson "
                    f"fallback is unavailable: {missing}",
                }
                return
            self._poisson(ctx, f"The height-map surface failed ({reason}).")
        run.timings["surface_processing_s"] = round(time.monotonic() - t0, 3)

    # -- height map + texture ---------------------------------------------------------------
    def _height_map(self, ctx: StageContext) -> None:
        run = self.run_state
        fused, final = run.fused, run.final
        assert fused is not None and final is not None
        n = len(fused.points)
        ctx.progress(f"Height map: fusing {n:,} points into a grid", None, None, None)
        cameras: list[heightfield.Camera] = []
        if run.dense is not None and run.dense.npz is not None and run.dense.transform is not None:
            views, records = _camera_records(run)
            for kf, rec in zip(views, records, strict=True):
                cameras.append(
                    heightfield.Camera(
                        c2w=np.asarray(rec["c2w"], dtype=np.float64),
                        fx=float(rec["fx"]),
                        fy=float(rec["fy"]),
                        cx=float(rec["cx"]),
                        cy=float(rec["cy"]),
                        image=run.images[kf],
                    )
                )

        def texture_progress(done: int, total: int) -> None:
            ctx.check_cancel()
            ctx.progress(
                f"Texture: projecting {len(cameras)} full-resolution frames onto the surface",
                done,
                total,
                "texture rows",
            )

        mesh, notes = heightfield.build_textured_heightfield(
            fused.points, fused.colors, final[2], cameras or None, on_progress=texture_progress
        )
        texture_bytes: bytes | None = None
        if mesh.texture is not None:
            ok, jpg = cv2.imencode(".jpg", mesh.texture[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
            if not ok:
                raise StageFailed("The surface texture could not be encoded.")
            texture_bytes = jpg.tobytes()
        mpu = run.frame().metres_per_unit
        k = mpu or 1.0
        grid_notes, tex_notes = notes["grid"], notes.get("texture")
        glb = write_mesh_glb(
            mesh.vertices.astype(np.float64) * k,
            mesh.faces,
            mesh.colors,
            mesh.normals,
            uv=mesh.uv if texture_bytes else None,
            texture=texture_bytes,
            texture_mime="image/jpeg",
            extras={
                "units": "metre" if mpu else "unitless",
                "scale_status": run.frame().scale_status.value,
                "frame": "local_reconstruction, +Y up",
                "method": "height-map (DSM) fusion"
                + (" with a texture baked from the video frames" if texture_bytes else ""),
                "closed_solid": mesh.closed,
                "note": "Top-down surface of what the video saw; see the processing report.",
            },
        )
        ctx.publish(
            _declare(
                run,
                "surface-mesh",
                ArtifactKind.MESH,
                "model/gltf-binary",
                glb,
                xyz=mesh.vertices.astype(np.float64),
                frames=_frames(run, run.placed()),
                final=True,
                spatial=True,
                stage="surface_processing",
            ),
            glb,
        )
        top = mesh.labels[: int(mesh.stats["surface_vertices"])]
        total = max(1, len(top))
        classes = {
            "observed": int((top == OBSERVED).sum()),
            "estimated": int((top == ESTIMATED).sum()),
            "inferred": int((top == heightfield.INFERRED).sum()),
        }
        limitations = [
            "Only what the video saw is on the surface: a single pass sees the top and the "
            "filmed side, never the far side, so the model is a relief with a flat base, not a "
            "full 360-degree object.",
            "Vertical faces (walls, facades) become steep slopes in a height map and are "
            "stretched in the texture.",
        ]
        if texture_bytes is None:
            limitations.append(
                f"No texture: {notes.get('texture_error', 'unavailable')} The surface carries "
                "per-vertex colour from the fused points instead."
            )
        else:
            limitations.append(
                "Each texel is blended from the frames that see it best; exposure differences "
                "between frames can still leave faint seams."
            )
        run.contributions["surface_processing"] = {
            "applied": True,
            "method": "height-map (DSM) fusion, solid block, planar texture",
            "what": "A smooth, closed, textured surface: points are fused into a height map "
            "(outliers removed, small holes bridged and labelled inferred), closed into a solid "
            "with side walls and a base, and textured from the full-resolution frames.",
            "grid": grid_notes,
            "texture": tex_notes
            if tex_notes
            else {"applied": False, "reason": notes.get("texture_error")},
            "mesh": notes["mesh"],
            "closed_solid": mesh.closed,
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "vertex_classes": classes,
            "inferred_vertex_share": round(classes["inferred"] / total, 4),
            "units": "metre" if mpu else "unitless",
            "limitations": limitations,
        }
        tail = ", textured" if texture_bytes else ", vertex-coloured (no texture)"
        ctx.progress(
            f"Surface: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles{tail}, "
            f"{100 * classes['inferred'] / total:.1f}% inferred",
            None,
            None,
            None,
        )

    # -- fallback -------------------------------------------------------------------------------
    def _poisson(self, ctx: StageContext, why: str) -> None:
        run = self.run_state
        fused, final = run.fused, run.final
        assert fused is not None and final is not None
        work = run.work("surface")
        np.savez(
            work / "cloud.npz",
            points=fused.points,
            colors=fused.colors,
            labels=final[2],
            cameras=fused.cameras,
            normals=fused.normals,
        )
        ctx.progress(
            f"Open3D: normals and screened Poisson (depth {run.profile.poisson_depth}) on "
            f"{len(fused.points):,} points",
            None,
            None,
            None,
        )
        done = self._gpu_step(
            ctx,
            "surface_processing",
            "Open3D surface reconstruction",
            "surface_runner.py",
            [
                "--cloud",
                str(work / "cloud.npz"),
                "--out",
                str(work / "mesh.npz"),
                "--depth",
                str(run.profile.poisson_depth),
            ],
            {},
        )
        if done is None:
            return
        mesh = np.load(work / "mesh.npz")
        mpu = run.frame().metres_per_unit
        k = mpu or 1.0
        glb = write_mesh_glb(
            mesh["vertices"] * k,
            mesh["faces"],
            mesh["colors"],
            mesh["normals"],
            extras={
                "units": "metre" if mpu else "unitless",
                "scale_status": run.frame().scale_status.value,
                "frame": "local_reconstruction, +Y up",
                "vertex_classes": done["vertex_classes"],
                "note": "Screened Poisson surface from NAZAR's fused cloud; see the report.",
            },
        )
        ctx.publish(
            _declare(
                run,
                "surface-mesh",
                ArtifactKind.MESH,
                "model/gltf-binary",
                glb,
                xyz=mesh["vertices"].astype(np.float64),
                frames=_frames(run, run.placed()),
                final=True,
                spatial=True,
                stage="surface_processing",
            ),
            glb,
        )
        classes = done["vertex_classes"]
        total = max(1, sum(classes.values()))
        run.contributions["surface_processing"] = {
            "applied": True,
            "method": "screened Poisson (fallback)",
            "what": "A coloured triangle mesh from screened Poisson on the fused cloud; surface "
            "not supported by any point is trimmed, and surface bridging gaps is labelled "
            "inferred. " + why,
            **{k2: v for k2, v in done.items() if k2 not in ("event", "timings")},
            "inferred_vertex_share": round(classes["inferred"] / total, 4),
            "units": "metre" if mpu else "unitless",
            "limitations": [
                "Poisson turns noise in the point cloud into lumps and has no texture.",
            ],
        }
        ctx.progress(
            f"Mesh: {done['vertices']:,} vertices, {done['triangles']:,} triangles, "
            f"{100 * classes['inferred'] / total:.1f}% inferred",
            None,
            None,
            None,
        )


def _statistical_inliers(points: npt.NDArray[np.float64], k: int = 8, sigma: float = 2.5) -> Any:
    if len(points) <= k:
        return np.ones(len(points), dtype=bool)
    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean = dist[:, 1:].mean(axis=1)
    return mean <= mean.mean() + sigma * mean.std()


class Analysis(_Stage):
    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        observed = run.observed_mask()
        n_obs = int(observed.sum())
        ctx.progress(
            f"{n_obs:,} observed / {len(observed) - n_obs:,} estimated points",
            len(observed),
            len(observed),
            "points",
        )
        al = run.alignment
        if al is not None and al.scale_accepted:
            ctx.warn(
                "scale_from_gnss",
                f"Scale comes from GNSS telemetry (1 unit = {al.metres_per_unit:.3f} m, residual "
                f"{al.residual_rms_m:.1f} m RMS, approx. +/-"
                f"{100 * (al.scale_uncertainty_rel or 0):.1f}% from fit noise alone). Consumer "
                "GNSS errors and clock sync limit it; treat distances as estimates.",
                "info",
            )
            if not al.orientation_accepted:
                ctx.warn("not_georeferenced", " ".join(al.reasons), "info")
        else:
            tail = (
                "distances are relative (unitless). Absolute size, height and geographic "
                "position are unknown without usable telemetry or a known reference length."
            )
            message = (
                f"Video only: {tail}"
                if al is None
                else f"Telemetry was supplied but not applied. {' '.join(al.reasons)} So {tail}"
            )
            ctx.warn("scale_unknown", message, "warning")
        applied = [k for k, c in run.contributions.items() if c.get("applied")]
        ctx.progress(f"Models that contributed: {', '.join(applied) or 'none'}", None, None, None)
        if run.method == FAST3R:
            ctx.warn(
                "learned_geometry",
                "Geometry is predicted by Fast3R, a learned multi-view model: dense, but "
                "predicted. Confidence is the model's own score, not a measured error.",
                "info",
            )
        else:
            ctx.warn(
                "sparse_baseline",
                "Geometry comes from the CPU baseline (classical structure-from-motion): a "
                "sparse cloud of matched features, not a dense surface.",
                "info",
            )


class Exporting(_Stage):
    def run(self, ctx: StageContext) -> None:
        run = self.run_state
        assert run.keep is not None
        observed = run.observed_mask()
        n_obs = int(observed.sum())
        if run.dense is not None:
            geometry: dict[str, Any] = {
                "method": MODELS[FAST3R],
                "model_version": _provenance(FAST3R).model_version,
                "licence": "FAIR Noncommercial Research License (academic use approved)",
                "placed_keyframes": len(run.dense.views),
                "points_before_filter": len(run.keep),
                "points": int(run.keep.sum()),
                "focal_px_estimated": round(run.dense.focal_px, 2),
                "focal_note": "Estimated by Fast3R's global head; not calibrated.",
                **{k: v for k, v in run.gpu_stats.items() if k not in ("device",)},
            }
            rule = (
                f"Pixels with Fast3R confidence below max({fast3r.KEEP_MIN_CONF}, the job's "
                f"{fast3r.KEEP_QUANTILE:.0f}th percentile) are dropped; kept points with "
                f"confidence >= {fast3r.OBSERVED_MIN_CONF} are observed, the rest estimated. "
                "Heuristic labels from the model's own score, not calibrated probabilities."
            )
            resources = {
                "device": run.gpu_stats.get("device") or "GPU",
                "peak_vram_mib": run.gpu_stats.get("peak_vram_mib"),
                "peak_host_commit_mib": run.gpu_stats.get("peak_host_commit_mib"),
                "host": platform.platform(),
            }
            limitations = [
                "Geometry is predicted by a learned model; thin structures and areas seen in "
                "few keyframes can be distorted.",
                "Unknown absolute scale; measurements are relative only.",
                "Up direction is estimated from the dominant plane and may be wrong for "
                "scenes without a ground plane.",
            ]
        else:
            snap = run.snapshot
            assert snap is not None
            geometry = {
                # Reconstruction stats first: the explicit fields below must win over them
                # (the final cloud is the filtered one, not the raw SfM output).
                **{k: v for k, v in snap.stats.items() if k not in ("points", "registered")},
                "method": MODELS[SFM],
                "registered_keyframes": len(snap.registered),
                "points_before_filter": len(snap.points),
                "points": int(run.keep.sum()),
                "focal_px_estimated": round(snap.focal_px, 2),
                "focal_note": "Not calibrated: initial guess refined by bundle adjustment.",
            }
            if run.fallback_reason:
                geometry["fast3r_fallback_reason"] = run.fallback_reason
            rule = (
                f"observed = seen by >= {OBSERVED_MIN_VIEWS} keyframes with mean reprojection "
                f"<= {OBSERVED_MAX_PX} px; estimated = everything else. Heuristic labels, not "
                "calibrated probabilities."
            )
            resources = {
                "device": "CPU",
                "peak_vram": "not applicable (CPU only)",
                "host": platform.platform(),
            }
            limitations = [
                "Sparse points only; textureless or repetitive surfaces are missing.",
                "Unknown absolute scale; measurements are relative only.",
                "Up direction is estimated from the dominant plane and may be wrong for "
                "scenes without a ground plane.",
            ]
        report: dict[str, Any] = {
            "schema": "nazar.processing-report/1",
            "generated_at": datetime.now(UTC).isoformat(),
            "job_id": ctx.lease.job_id,
            "attempt": ctx.lease.attempt,
            "pipeline": "standard",
            "geometry_method": run.method,
            "input": {
                "video_sha256": ctx.lease.video_sha256,
                "keyframes": len(run.images),
                "telemetry": run.telemetry,
                "image_size": list(run.images[0].shape[1::-1]) if run.images else None,
            },
            "geometry": geometry,
            "confidence": {
                "observed": n_obs,
                "estimated": len(observed) - n_obs,
                "inferred": 0,
                "rule": rule,
            },
            "coordinate_frame": {
                "type": "local_reconstruction",
                "axes": "right-handed, +Y up (estimated from the dominant plane), -Z forward",
                "units": "unitless",
                "scale_status": run.frame().scale_status.value,
                "metres_per_unit": run.frame().metres_per_unit,
                "normalisation": "median camera distance from the scene centre = 10 units",
            },
            "timings_s": {**run.timings, "first_preview_s": run.first_preview_s},
            "resources": resources,
            "profile": run.profile.report(),
            "contributions": run.contributions,
            "skipped": {
                k: str(c.get("reason"))
                for k, c in run.contributions.items()
                if not c.get("applied")
            },
            "limitations": limitations,
        }
        budget = run.profile.vram_budget_mib
        resources["gpu_peak_vram_mib_by_stage"] = run.gpu_peaks
        resources["gpu_budget_mib"] = budget or None
        resources["within_gpu_budget"] = (
            all(v <= budget for v in run.gpu_peaks.values()) if budget else None
        )
        mesh = run.contributions.get("surface_processing", {})
        if mesh.get("applied"):
            report["confidence"]["mesh_vertices"] = mesh["vertex_classes"]
        if run.telemetry is not None:
            report["sensor_alignment"] = report["input"].pop("telemetry")
            report["input"]["telemetry"] = True
            if run.frame().scale_status is ScaleStatus.METRIC_SENSOR:
                limitations[1] = (
                    "Scale comes from consumer GNSS; distances are estimates (see "
                    "sensor_alignment)."
                )
        assert run.final is not None
        pts, cols, labels = run.final
        al = run.alignment
        if al is not None and al.orientation_accepted:
            utm, code = al.to_utm(pts)
            las = write_las(
                utm, cols, labels, enu=True, epsg=code, system_id=b"NAZAR UTM from GNSS alignment"
            )
            assert run.telemetry is not None
            run.telemetry["crs"] = f"EPSG:{code}"
            las_note = (
                f"LAS 1.2, point format 2 (RGB). Horizontal: EPSG:{code} (WGS 84 / UTM zone "
                f"{code % 100}{'N' if code < 32700 else 'S'}), declared in a GeoKey VLR. "
                f"Heights in metres, {run.telemetry['altitude_kind']}; no vertical datum is "
                "claimed. Placement inherits consumer-GNSS error (metres)."
            )
        elif al is not None and al.scale_accepted:
            assert al.metres_per_unit is not None
            las = write_las(
                pts * al.metres_per_unit, cols, labels, system_id=b"NAZAR local frame m, no CRS"
            )
            las_note = (
                "LAS 1.2, point format 2 (RGB). Local frame in metres (GNSS scale), not "
                "georeferenced. Axes converted to +Z up: LAS (X, Y, Z) = NAZAR (x, -z, y)."
            )
        else:
            las = write_las(pts, cols, labels)
            las_note = (
                "LAS 1.2, point format 2 (RGB). No CRS: local frame, unknown scale. Axes "
                "converted to +Z up: LAS (X, Y, Z) = NAZAR (x, -z, y)."
            )
        ctx.publish(
            _declare(
                run,
                "export-las",
                ArtifactKind.EXPORT,
                "application/vnd.las",
                las,
                xyz=pts,
                frames=_frames(run, run.placed()),
                final=True,
            ),
            las,
        )
        # glTF assumes metres and +Y up (NAZAR's axes): scale applies only when it is known.
        mpu = run.frame().metres_per_unit
        k = mpu or 1.0
        glb = write_glb(
            pts * k,
            cols,
            run.centres() * k,
            extras={
                "units": "metre" if mpu else "unitless",
                "scale_status": run.frame().scale_status.value,
                "frame": "local_reconstruction, +Y up",
                "note": "Point cloud and camera path from NAZAR; see the processing report.",
            },
        )
        ctx.publish(
            _declare(
                run,
                "export-glb",
                ArtifactKind.EXPORT,
                "model/gltf-binary",
                glb,
                xyz=pts,
                frames=_frames(run, run.placed()),
                final=True,
            ),
            glb,
        )
        report["exports"] = {
            "las": f"{las_note} Classification 1 (unclassified); user data = confidence class "
            "(1 observed, 2 estimated).",
            "glb": "glTF 2.0 binary: the final cloud as POINTS (COLOR_0 sRGB) and the camera path "
            "as a LINE_STRIP, +Y up. "
            + (
                f"Coordinates in metres ({run.frame().scale_status.value} scale), local frame."
                if mpu
                else "Coordinates unitless (unknown scale), although glTF viewers assume metres."
            ),
        }
        if mesh.get("applied"):
            report["exports"]["mesh"] = (
                "glTF 2.0 binary surface-mesh: indexed TRIANGLES with NORMAL and COLOR_0, same "
                "frame and units as the GLB export; per-vertex classes counted in "
                "confidence.mesh_vertices."
            )
        if run.contributions.get("gaussian_splat", {}).get("applied"):
            report["exports"]["gaussian_scene"] = (
                "3D Gaussian splat in the standard 3DGS PLY layout (x, y, z, f_dc, f_rest, "
                "opacity, scale, rot), NAZAR frame and units; for viewing, not measuring."
            )
        content = json.dumps(report, indent=2).encode()
        ctx.publish(
            _declare(run, "report", ArtifactKind.REPORT, "application/json", content, final=True),
            content,
        )


class Skip:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def skip_reason(self, lease: WorkerLease) -> str | None:
        return self.reason

    def run(self, ctx: StageContext) -> None:
        raise AssertionError("a skipped stage never runs")


def build_standard_pipeline(
    runtime: Fast3rRuntime | str | None = None,
    *,
    profile: ComputeProfile = LOCAL,
    gpu: GpuRuntime | None = None,
    work_dir: Path | None = None,
) -> dict[StageName, StageAdapter]:
    """`runtime`: the Fast3R runtime, or why it is unavailable; read from the environment
    when omitted. `profile`: local (6 GB budget) or clg. `work_dir`: keep model inputs and
    outputs there (the local test script does; workers use a temporary directory)."""
    run = _Run(
        runtime=Fast3rRuntime.from_env() if runtime is None else runtime,
        profile=profile,
        gpu=GpuRuntime.from_env() if gpu is None else gpu,
        work_dir=work_dir,
    )
    return {
        StageName.PREPROCESSING: Preprocessing(run),
        StageName.DYNAMIC_ANALYSIS: DynamicAnalysis(run),
        StageName.GEOMETRY: Geometry(run),
        StageName.DEPTH_PRIORS: DepthPriors(run),
        StageName.SENSOR_OPTIMIZATION: SensorOptimization(run),
        StageName.DENSE_RECONSTRUCTION: Densify(run),
        StageName.SURFACE_PROCESSING: Surface(run),
        StageName.ANALYSIS: Analysis(run),
        StageName.EXPORTING: Exporting(run),
    }
