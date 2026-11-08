"""Job lifecycle state machine.

Processing stages run in a fixed order. A stage may be *skipped* (for example sensor
optimisation when no telemetry was supplied); a skip is reported with a `stage.skipped`
event and a reason, and is never reported as a successful completion.
"""

from enum import StrEnum


class JobStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    DYNAMIC_ANALYSIS = "dynamic_analysis"
    GEOMETRY = "geometry"
    DEPTH_PRIORS = "depth_priors"
    SENSOR_OPTIMIZATION = "sensor_optimization"
    DENSE_RECONSTRUCTION = "dense_reconstruction"
    SURFACE_PROCESSING = "surface_processing"
    ANALYSIS = "analysis"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class ComputeTarget(StrEnum):
    """Where a job's GPU work runs. Each worker declares its location and only leases jobs
    aimed at it (the API never moves a job between locations on its own)."""

    LOCAL = "local"
    COLLEGE = "college"


FPS_MIN = 0.25
FPS_MAX = 4.0
FPS_DEFAULT = 1.0
"""Frames per second of video the reconstruction samples (the user's choice, default 1)."""
MIN_FRAMES = 3
"""Fewer frames cannot be reconstructed."""
MAX_FRAMES: dict[ComputeTarget, int] = {ComputeTarget.LOCAL: 64, ComputeTarget.COLLEGE: 128}
"""Most frames one job may use at each location. The local cap keeps a 6 GB card's run time
and memory in check; a request above it is lowered (and reported), never silently cut."""
AUTO_MIN_FRAMES = 24
"""Automatic mode never uses fewer frames than this (a short clip is sampled more densely
than 1 fps): a handful of views cannot give a usable reconstruction."""
AUTO_MAX_FRAMES: dict[ComputeTarget, int] = {ComputeTarget.LOCAL: 40, ComputeTarget.COLLEGE: 96}
"""Most frames automatic mode asks for. Below the hard caps above so that a long clip still
finishes in reasonable time and memory on that location."""


class StageName(StrEnum):
    """Processing stages. Each value is also the job status while that stage runs.

    Moving objects are masked *before* geometry, so the masks keep them out of the pose
    estimate as well as out of the points."""

    PREPROCESSING = "preprocessing"
    DYNAMIC_ANALYSIS = "dynamic_analysis"
    GEOMETRY = "geometry"
    DEPTH_PRIORS = "depth_priors"
    SENSOR_OPTIMIZATION = "sensor_optimization"
    DENSE_RECONSTRUCTION = "dense_reconstruction"
    SURFACE_PROCESSING = "surface_processing"
    ANALYSIS = "analysis"
    EXPORTING = "exporting"


PIPELINE_STAGES: tuple[StageName, ...] = tuple(StageName)

TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)

_STAGE_STATUSES: tuple[JobStatus, ...] = tuple(JobStatus(stage.value) for stage in PIPELINE_STAGES)


def _build_transitions() -> dict[JobStatus, frozenset[JobStatus]]:
    t: dict[JobStatus, set[JobStatus]] = {
        # Inputs are validated at upload, so a new job usually goes straight to the queue.
        JobStatus.CREATED: {JobStatus.VALIDATING, JobStatus.QUEUED},
        JobStatus.VALIDATING: {JobStatus.QUEUED},
        JobStatus.QUEUED: {_STAGE_STATUSES[0]},
        JobStatus.CANCEL_REQUESTED: {JobStatus.CANCELLED, JobStatus.FAILED},
        # Retry re-queues a finished-but-unsuccessful job; completed work is reused from cache.
        JobStatus.FAILED: {JobStatus.QUEUED},
        JobStatus.CANCELLED: {JobStatus.QUEUED},
        JobStatus.COMPLETED: set(),
    }
    for i, status in enumerate(_STAGE_STATUSES):
        # Any later stage is reachable: intermediate stages were skipped (with a reason).
        t[status] = set(_STAGE_STATUSES[i + 1 :])
        # A worker that stops renewing its lease hands the job back to the queue.
        t[status].add(JobStatus.QUEUED)
    t[_STAGE_STATUSES[-1]].add(JobStatus.COMPLETED)

    for status, targets in t.items():
        if status not in TERMINAL_STATUSES and status is not JobStatus.CANCEL_REQUESTED:
            targets.update({JobStatus.FAILED, JobStatus.CANCEL_REQUESTED})
    return {status: frozenset(targets) for status, targets in t.items()}


ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = _build_transitions()


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]
