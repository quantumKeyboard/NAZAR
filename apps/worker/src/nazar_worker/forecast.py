"""What a standard job can run here, and how long it should take.

`model_readiness` tells the UI which models this worker can run right now (and why not).
`Forecast` turns that, the frame count and the compute profile into a per-stage time estimate
that is sent as the `job.estimate` event. The estimate is honest about its source:

- the built-in cost model (a fixed part plus a part per frame, per stage) was fitted to a
  measured run on an RTX 4050 Laptop 6 GB (sample1, 22 frames, local profile, idle machine);
- for the college profile the same model is scaled and labelled as not yet measured;
- after every job the real stage durations are stored next to the GPU runtime, and later
  estimates use the median ratio of real to modelled time per stage, so they converge on
  this machine ("measured on this machine").
"""

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nazar_contracts import (
    PIPELINE_STAGES,
    JobEstimateData,
    ModelReadiness,
    StageName,
    TimeModel,
)
from nazar_worker.fast3r import Fast3rRuntime
from nazar_worker.gpu import GpuRuntime
from nazar_worker.profile import ComputeProfile

log = logging.getLogger("nazar_worker")

_DEPTH_WEIGHTS = {
    "vit_small": "metric_depth_vit_small_800k.pth",
    "vit_large": "metric_depth_vit_large_800k.pth",
    "vit_giant2": "metric_depth_vit_giant2_800k.pth",
}

HISTORY = 5
"""Runs remembered per stage; the median of them calibrates the next estimate."""


def model_readiness(profile: ComputeProfile, gpu: GpuRuntime) -> list[ModelReadiness]:
    """The models this worker can run under `profile`, in pipeline order."""

    def row(name: str, stage: StageName, missing: str | None, ok: str) -> ModelReadiness:
        return ModelReadiness(name=name, stage=stage, ready=missing is None, detail=missing or ok)

    fast3r = Fast3rRuntime.from_env()
    depth_missing = gpu.missing(
        *[f"metric3d/{_DEPTH_WEIGHTS[m]}" for m in profile.metric3d_models[:1]]
    )
    if depth_missing and not profile.strict:  # the smaller variant is an accepted fallback
        depth_missing = gpu.missing(
            *[f"metric3d/{_DEPTH_WEIGHTS[m]}" for m in profile.metric3d_models[-1:]]
        )
    return [
        row(
            "YOLO",
            StageName.DYNAMIC_ANALYSIS,
            gpu.missing(f"yolo/{profile.yolo_weights}"),
            profile.yolo_weights,
        ),
        row(
            "SAM 2",
            StageName.DYNAMIC_ANALYSIS,
            gpu.missing(f"sam2/{profile.sam2_checkpoint}"),
            profile.sam2_checkpoint,
        ),
        row(
            "Fast3R",
            StageName.GEOMETRY,
            fast3r if isinstance(fast3r, str) else None,
            "Fast3R ViT-L/512",
        ),
        row(
            "Metric3D-v2", StageName.DEPTH_PRIORS, depth_missing, "/".join(profile.metric3d_models)
        ),
        row(
            "gsplat", StageName.DENSE_RECONSTRUCTION, gpu.missing(splat=True), "Gaussian splatting"
        ),
        # The height-map surface and its texture run on the CPU in the worker process; Open3D
        # (GPU environment) is only the fallback, so it does not gate this stage.
        row("Height map", StageName.SURFACE_PROCESSING, None, "surface + texture (CPU)"),
    ]


# (fixed seconds, seconds per frame) fitted to the measured local run: sample1, 22 frames.
LOCAL_COST: dict[StageName, tuple[float, float]] = {
    StageName.PREPROCESSING: (0.5, 0.05),
    StageName.DYNAMIC_ANALYSIS: (3.0, 0.15),
    StageName.GEOMETRY: (6.0, 0.9),
    StageName.DEPTH_PRIORS: (7.0, 2.1),
    StageName.SENSOR_OPTIMIZATION: (0.3, 0.0),
    StageName.DENSE_RECONSTRUCTION: (3.0, 0.3),  # fusion; the splat is added by iteration count
    StageName.SURFACE_PROCESSING: (20.0, 3.0),  # height map + texture bake (CPU), per frame
    StageName.ANALYSIS: (0.2, 0.0),
    StageName.EXPORTING: (0.4, 0.03),
}
SPLAT_SETUP_S = 25.0
SPLAT_S_PER_ITERATION = 0.0645
CPU_BASELINE_COST = {StageName.GEOMETRY: (10.0, 1.6), StageName.DENSE_RECONSTRUCTION: (1.0, 0.1)}
COLLEGE_SPEEDUP = 0.4
"""Assumed GPU speed-up of the college card over the 4050 laptop (not measured)."""


@dataclass(frozen=True)
class Forecast:
    profile: ComputeProfile
    ready: dict[StageName, bool]
    """Per stage: can its model(s) run on this worker."""
    cpu_baseline: bool
    store: Path | None = None

    @staticmethod
    def for_worker(profile: ComputeProfile, gpu: GpuRuntime | None = None) -> "Forecast":
        gpu = gpu or GpuRuntime.from_env()
        models = model_readiness(profile, gpu)
        ready = {s: True for s in PIPELINE_STAGES}
        for m in models:
            ready[m.stage] = ready[m.stage] and m.ready
        fast3r = next(m for m in models if m.name == "Fast3R").ready
        return Forecast(profile, ready, cpu_baseline=not fast3r, store=gpu.root / "timings.json")

    @property
    def key(self) -> str:
        """History is kept apart per profile, and for the CPU baseline."""
        return self.profile.name + ("-cpu" if self.cpu_baseline else "")

    def modelled(self, stage: StageName, frames: int, telemetry: bool) -> float:
        """Built-in cost of `stage` for `frames` frames, 0 when it is known not to run."""
        if stage is StageName.SENSOR_OPTIMIZATION and not telemetry:
            return 0.0
        if self.cpu_baseline:
            if stage in (
                StageName.DYNAMIC_ANALYSIS,
                StageName.DEPTH_PRIORS,
                StageName.SURFACE_PROCESSING,
            ):
                return 0.0
            fixed, per = CPU_BASELINE_COST.get(stage, LOCAL_COST[stage])
            return fixed + per * frames
        if not self.ready[stage] and not self.profile.strict:
            return 0.0
        fixed, per = LOCAL_COST[stage]
        seconds = fixed + per * frames
        if stage is StageName.DENSE_RECONSTRUCTION and self.ready[stage]:
            seconds += SPLAT_SETUP_S + SPLAT_S_PER_ITERATION * self.profile.splat_iterations
        gpu_bound = {
            StageName.GEOMETRY,
            StageName.DEPTH_PRIORS,
            StageName.DYNAMIC_ANALYSIS,
            StageName.DENSE_RECONSTRUCTION,
        }
        if self.profile.strict and stage in gpu_bound:
            seconds *= COLLEGE_SPEEDUP
        return seconds

    # --- measured history ---------------------------------------------------------------

    def _history(self) -> dict[str, dict[str, list[dict[str, float]]]]:
        if self.store is None or not self.store.is_file():
            return {}
        try:
            data: dict[str, dict[str, list[dict[str, float]]]] = json.loads(
                self.store.read_text(encoding="utf-8")
            )
            return data
        except (OSError, ValueError):
            return {}

    def _ratio(self, stage: StageName, history: dict[str, Any], telemetry: bool) -> float | None:
        runs = history.get(self.key, {}).get(stage.value, [])
        ratios = []
        for run in runs[-HISTORY:]:
            # A stage that did different work then (no splat trained) says nothing about now.
            if bool(run.get("full", True)) != self._full(stage):
                continue
            predicted = self.modelled(stage, int(run["frames"]), telemetry)
            if predicted > 0 and run["seconds"] > 0:
                ratio = run["seconds"] / predicted
                if 0.2 <= ratio <= 5:  # outside this the model, not the machine, is wrong
                    ratios.append(ratio)
        return statistics.median(ratios) if ratios else None

    def _full(self, stage: StageName) -> bool:
        """Whether `stage` does all of its work here (dense: including the splat)."""
        return self.ready[stage] if stage is StageName.DENSE_RECONSTRUCTION else True

    def estimate(self, frames: int, telemetry: bool) -> JobEstimateData:
        history = self._history()
        stages: dict[StageName, float] = {}
        measured = 0
        for stage in PIPELINE_STAGES:
            base = self.modelled(stage, frames, telemetry)
            ratio = self._ratio(stage, history, telemetry) if base > 0 else None
            if ratio is not None:
                measured += 1
            stages[stage] = round(base * (ratio if ratio is not None else 1.0), 1)
        active = sum(1 for s in stages.values() if s > 0)
        if measured and measured >= active // 2:
            basis = f"Calibrated from earlier jobs on this machine ({measured} stages measured)."
        elif self.profile.strict:
            basis = (
                "Built-in estimate for the college GPU; not yet measured there, "
                "so it may be well off."
            )
        elif self.cpu_baseline:
            basis = "Built-in estimate for the CPU baseline (no GPU models are set up here)."
        else:
            basis = (
                "Built-in estimate from a measured run on an RTX 4050 laptop; "
                "it improves after your first job."
            )
        return JobEstimateData(
            frames=frames, basis=basis, stages=stages, total_s=round(sum(stages.values()), 1)
        )

    def time_model(self) -> TimeModel:
        """Total time as a line in the frame count, fitted through two estimates."""
        a, b = self.estimate(10, False), self.estimate(40, False)
        per_frame = max(0.0, (b.total_s - a.total_s) / 30)
        return TimeModel(
            fixed_s=round(max(0.0, a.total_s - 10 * per_frame), 1),
            per_frame_s=round(per_frame, 2),
            basis=b.basis,
        )

    def record(self, stage: StageName, frames: int, seconds: float) -> None:
        """Remember how long `stage` really took (completed stages only)."""
        if self.store is None or seconds <= 0:
            return
        history = self._history()
        runs = history.setdefault(self.key, {}).setdefault(stage.value, [])
        runs.append({"frames": frames, "seconds": round(seconds, 2), "full": self._full(stage)})
        del runs[:-HISTORY]
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_suffix(".tmp")
            tmp.write_text(json.dumps(history, indent=1), encoding="utf-8")
            tmp.replace(self.store)
        except OSError as exc:
            log.warning("could not save stage timings: %s", exc)
