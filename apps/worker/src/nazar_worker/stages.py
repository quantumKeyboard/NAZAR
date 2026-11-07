"""Stage adapter boundary.

Every pipeline stage is implemented by an adapter that reads declared inputs from the job
workspace and writes declared, checksummed outputs atomically. Adapters never touch the
database or the browser; the worker loop turns their results into contract events.

`standard.build_standard_pipeline` builds the standard pipeline (Fast3R or CPU SfM) and
`test_pipeline.build_test_pipeline` the deterministic test pipeline used by CI and demos.
`STAGE_PLAN` records the intended component and the milestone that delivers each one,
so the UI and docs can state status honestly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from nazar_contracts import ArtifactCreate, ArtifactCreated, StageName, WorkerLease

WarningSeverity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class StagePlan:
    stage: StageName
    component: str
    milestone: int
    optional: bool


STAGE_PLAN: tuple[StagePlan, ...] = (
    StagePlan(StageName.PREPROCESSING, "FFmpeg + OpenCV", 2, optional=False),
    StagePlan(StageName.DYNAMIC_ANALYSIS, "YOLO11 + SAM 2 (masks before geometry)", 6, True),
    StagePlan(StageName.GEOMETRY, "Fast3R (GPU); CPU baseline: OpenCV SfM", 3, optional=False),
    StagePlan(StageName.DEPTH_PRIORS, "Metric3D-v2", 5, optional=True),
    StagePlan(
        StageName.SENSOR_OPTIMIZATION, "Sim(3) GNSS alignment (telemetry only)", 5, optional=True
    ),
    StagePlan(
        StageName.DENSE_RECONSTRUCTION, "Fusion + Gaussian splat (gsplat); CPU: sparse", 7, False
    ),
    StagePlan(
        StageName.SURFACE_PROCESSING, "Height-map solid with baked texture", 7, optional=True
    ),
    StagePlan(StageName.ANALYSIS, "Confidence + per-model contributions", 8, optional=False),
    StagePlan(StageName.EXPORTING, "PLY / LAS / GLB / mesh / splat + report", 8, optional=False),
)


class Cancelled(Exception):
    """Raised inside a stage when the user asked to cancel; the runner reports it."""


class StageFailed(Exception):
    """The input cannot produce this stage's result. The message is shown to the user, and
    retrying the same input would fail the same way unless `retryable` says otherwise."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class StageContext:
    """What an adapter may do. It never talks to the API or the database directly."""

    lease: WorkerLease
    progress: Callable[[str, int | None, int | None, str | None], None]
    warn: Callable[[str, str, WarningSeverity], None]
    publish: Callable[[ArtifactCreate, bytes], ArtifactCreated]
    preflight: Callable[[], dict[str, Any]]
    keyframe: Callable[[int], bytes]
    """JPEG bytes of a preflight keyframe, by its 1-based image number."""
    check_cancel: Callable[[], None]
    telemetry: Callable[[], dict[str, Any] | None] = lambda: None
    """The job's normalised telemetry track, or None (not supplied, or removed since)."""
    gpu: Callable[[int | None, int | None], None] = lambda peak, budget: None
    """Report the GPU memory a stage used: (peak MiB, budget MiB; 0 = no cap). The runner adds
    the largest peak to the stage's `stage.completed` event."""


class StageAdapter(Protocol):
    def skip_reason(self, lease: WorkerLease) -> str | None:
        """Why this stage cannot meaningfully run on this job's inputs, or None to run it.
        Asked before the stage starts: a started stage is never reported as skipped."""
        ...

    def run(self, ctx: StageContext) -> None:
        """Run the stage. Raise to fail the job."""
        ...
