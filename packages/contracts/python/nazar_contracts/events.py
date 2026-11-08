"""Job event stream contract (WebSocket + REST replay).

Every event carries the job id, a per-job monotonically increasing `seq` cursor, a
timestamp, the job status after the event, and the stage it concerns. Clients resume
by asking for events with `seq` greater than the last one they applied.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from nazar_contracts.artifacts import ArtifactManifest
from nazar_contracts.jobs import JobStatus, StageName

EVENT_SCHEMA_VERSION: Literal[1] = 1


class EventType(StrEnum):
    JOB_CREATED = "job.created"
    JOB_QUEUED = "job.queued"
    STAGE_STARTED = "stage.started"
    STAGE_PROGRESS = "stage.progress"
    STAGE_COMPLETED = "stage.completed"
    STAGE_SKIPPED = "stage.skipped"
    JOB_ESTIMATE = "job.estimate"
    WARNING = "warning"
    ARTIFACT_AVAILABLE = "artifact.available"
    PREVIEW_UPDATED = "preview.updated"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    CANCEL_REQUESTED = "cancel.requested"
    JOB_CANCELLED = "job.cancelled"


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = EVENT_SCHEMA_VERSION
    job_id: str
    seq: int = Field(ge=1)
    ts: datetime
    status: JobStatus
    stage: StageName | None = None


class JobCreatedData(BaseModel):
    mission_id: str
    test_pipeline: bool


class JobCreated(_Event):
    type: Literal[EventType.JOB_CREATED]
    data: JobCreatedData


class JobQueuedData(BaseModel):
    attempt: int = Field(ge=1)
    reason: str = Field(description="Why the job (re)entered the queue, e.g. retry or lost worker.")


class JobQueued(_Event):
    """The job went back to the queue (user retry, or a worker stopped renewing its lease)."""

    type: Literal[EventType.JOB_QUEUED]
    data: JobQueuedData


class StageStarted(_Event):
    type: Literal[EventType.STAGE_STARTED]


class StageProgressData(BaseModel):
    message: str
    completed: int | None = Field(
        default=None, ge=0, description="Units of real work done. Null when not measurable."
    )
    total: int | None = Field(
        default=None, ge=0, description="Total units. Null means indeterminate progress."
    )
    unit: str | None = Field(default=None, description="e.g. 'frames', 'keyframes', 'iterations'")


class StageProgress(_Event):
    type: Literal[EventType.STAGE_PROGRESS]
    data: StageProgressData


class StageCompletedData(BaseModel):
    duration_s: float = Field(ge=0)
    peak_vram_mib: int | None = Field(
        default=None, ge=0, description="Peak GPU memory this stage's models used, if any."
    )
    budget_mib: int | None = Field(
        default=None, ge=0, description="The GPU memory budget it ran under (0: unlimited)."
    )


class StageCompleted(_Event):
    type: Literal[EventType.STAGE_COMPLETED]
    data: StageCompletedData


class JobEstimateData(BaseModel):
    """The worker's forecast of how long each stage will take, sent once the job is leased.
    Clients combine it with the stages' real progress to show the time remaining."""

    frames: int = Field(ge=0, description="Frames the estimate was made for.")
    basis: str = Field(description="Where the numbers come from, in words (shown to the user).")
    stages: dict[StageName, float] = Field(
        description="Estimated seconds per stage; 0 for a stage already known to be skipped."
    )
    total_s: float = Field(ge=0)


class JobEstimate(_Event):
    type: Literal[EventType.JOB_ESTIMATE]
    data: JobEstimateData


class StageSkippedData(BaseModel):
    reason: str


class StageSkipped(_Event):
    type: Literal[EventType.STAGE_SKIPPED]
    data: StageSkippedData


class WarningData(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "critical"]


class WarningRaised(_Event):
    type: Literal[EventType.WARNING]
    data: WarningData


class ArtifactAvailable(_Event):
    type: Literal[EventType.ARTIFACT_AVAILABLE]
    data: ArtifactManifest


class PreviewUpdatedData(BaseModel):
    artifact_id: str
    revision: int = Field(ge=1)


class PreviewUpdated(_Event):
    type: Literal[EventType.PREVIEW_UPDATED]
    data: PreviewUpdatedData


class JobCompletedData(BaseModel):
    artifact_ids: list[str]


class JobCompleted(_Event):
    type: Literal[EventType.JOB_COMPLETED]
    data: JobCompletedData


class JobFailedData(BaseModel):
    reason: str = Field(description="Human-readable, actionable failure reason.")
    retryable: bool


class JobFailed(_Event):
    type: Literal[EventType.JOB_FAILED]
    data: JobFailedData


class CancelRequested(_Event):
    type: Literal[EventType.CANCEL_REQUESTED]


class JobCancelled(_Event):
    type: Literal[EventType.JOB_CANCELLED]


JobEvent = Annotated[
    JobCreated
    | JobQueued
    | StageStarted
    | StageProgress
    | StageCompleted
    | StageSkipped
    | JobEstimate
    | WarningRaised
    | ArtifactAvailable
    | PreviewUpdated
    | JobCompleted
    | JobFailed
    | CancelRequested
    | JobCancelled,
    Field(discriminator="type"),
]
