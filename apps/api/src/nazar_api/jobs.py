"""Job lifecycle: the append-only event log, status derivation, snapshots and worker leases.

Every state change is an event. `append_event` validates it against the contract and the
state machine, assigns the next `seq` atomically and caches the resulting status on the job
row. Snapshots are rebuilt from the log, so what a reconnecting client sees always agrees
with what a live subscriber was sent.
"""

import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from nazar_api.models import ArtifactRecord, JobEventRecord, JobRecord, utcnow
from nazar_contracts import (
    MAX_FRAMES,
    PIPELINE_STAGES,
    TERMINAL_STATUSES,
    ArtifactManifest,
    ComputeTarget,
    EventType,
    Job,
    JobEstimateData,
    JobEvent,
    JobStatus,
    PipelineMode,
    SamplingInfo,
    StageName,
    StageProgressData,
    StageRunState,
    StageState,
    WarningData,
    can_transition,
)

EVENTS: TypeAdapter[JobEvent] = TypeAdapter(JobEvent)

STAGE_ORDER = {stage: i for i, stage in enumerate(PIPELINE_STAGES)}

# Events a worker may report. The API itself emits job.created/queued, cancel.requested and
# artifact.available/preview.updated (the latter only once uploaded bytes are verified).
WORKER_EVENT_TYPES = frozenset(
    {
        EventType.STAGE_STARTED,
        EventType.STAGE_PROGRESS,
        EventType.STAGE_COMPLETED,
        EventType.STAGE_SKIPPED,
        EventType.JOB_ESTIMATE,
        EventType.WARNING,
        EventType.JOB_COMPLETED,
        EventType.JOB_FAILED,
        EventType.JOB_CANCELLED,
    }
)

_STATUS_FOR_EVENT = {
    EventType.JOB_QUEUED: JobStatus.QUEUED,
    EventType.JOB_COMPLETED: JobStatus.COMPLETED,
    EventType.JOB_FAILED: JobStatus.FAILED,
    EventType.CANCEL_REQUESTED: JobStatus.CANCEL_REQUESTED,
    EventType.JOB_CANCELLED: JobStatus.CANCELLED,
}


class EventRejected(Exception):
    """The event is malformed or not allowed in the job's current state."""


def _stage_status(status: JobStatus) -> StageName | None:
    try:
        return StageName(status.value)
    except ValueError:
        return None


def _next_status(current: JobStatus, type_: EventType, stage: StageName | None) -> JobStatus:
    if type_ is EventType.JOB_CREATED:
        if current is not JobStatus.CREATED:
            raise EventRejected("job.created can only be the first event.")
        return current
    if type_ in _STATUS_FOR_EVENT:
        target = _STATUS_FOR_EVENT[type_]
    elif type_ is EventType.STAGE_STARTED:
        if stage is None:
            raise EventRejected("stage.started needs a stage.")
        if current is JobStatus.CANCEL_REQUESTED:
            raise EventRejected("Cancellation was requested; do not start new stages.")
        target = JobStatus(stage.value)
    else:
        running = _stage_status(current)
        # While cancellation is pending the running stage may still report and finish.
        if type_ in (EventType.STAGE_PROGRESS, EventType.STAGE_COMPLETED) and (
            stage is None or (running != stage and current is not JobStatus.CANCEL_REQUESTED)
        ):
            raise EventRejected(f"{type_.value} must concern the running stage.")
        if type_ is EventType.STAGE_SKIPPED:
            if stage is None:
                raise EventRejected("stage.skipped needs a stage.")
            if running is not None and STAGE_ORDER[stage] <= STAGE_ORDER[running]:
                raise EventRejected("Only a stage that has not started yet can be skipped.")
        return current
    if not can_transition(current, target):
        raise EventRejected(f"Cannot go from {current.value} to {target.value}.")
    return target


def append_event(
    db: Session,
    job: JobRecord,
    type_: EventType,
    *,
    stage: StageName | None = None,
    data: dict[str, Any] | None = None,
    key: str | None = None,
) -> dict[str, Any] | None:
    """Validate and append one event. Returns the stored payload, or None when `key` was
    already recorded (idempotent redelivery). The caller commits."""
    if key is not None:
        dup = db.scalar(
            select(JobEventRecord.seq).where(
                JobEventRecord.job_id == job.id, JobEventRecord.key == key
            )
        )
        if dup is not None:
            return None
    current = JobStatus(job.status)
    status = _next_status(current, type_, stage)

    # Atomic increment: concurrent writers can never be handed the same seq.
    seq = db.execute(
        update(JobRecord)
        .where(JobRecord.id == job.id)
        .values(last_seq=JobRecord.last_seq + 1)
        .returning(JobRecord.last_seq)
    ).scalar_one()
    raw: dict[str, Any] = {
        "job_id": job.id,
        "seq": seq,
        "ts": utcnow().isoformat(),
        "status": status.value,
        "stage": stage.value if stage else None,
        "type": type_.value,
    }
    if data is not None:
        raw["data"] = data
    try:
        payload: dict[str, Any] = EVENTS.dump_python(EVENTS.validate_python(raw), mode="json")
    except ValidationError as exc:
        db.rollback()
        raise EventRejected(_first_error(exc)) from exc
    db.add(JobEventRecord(job_id=job.id, seq=seq, key=key, payload=payload))
    job.last_seq = seq
    job.status = status.value
    if status in TERMINAL_STATUSES:
        job.lease_id = job.lease_worker = job.lease_expires_at = None
    return payload


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err["loc"][1:]) or "event"
    return f"Invalid event ({where}): {err['msg']}"


def events_after(db: Session, job_id: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(JobEventRecord.payload)
        .where(JobEventRecord.job_id == job_id, JobEventRecord.seq > after)
        .order_by(JobEventRecord.seq)
        .limit(limit)
    ).all()
    return list(rows)


def to_manifest(row: ArtifactRecord) -> ArtifactManifest:
    declared = {k: v for k, v in row.declared.items() if k not in ("key", "final")}
    return ArtifactManifest.model_validate(
        {**declared, "id": row.id, "job_id": row.job_id, "revision": row.revision}
        | {"created_at": row.created_at}
    )


def current_artifacts(db: Session, job: JobRecord) -> list[ArtifactManifest]:
    """Newest available revision of each kind (and media type) from the current attempt.
    Revisions of one output replace each other (progressive previews); different formats of
    one kind (LAS and GLB exports) are separate outputs."""
    rows = db.scalars(
        select(ArtifactRecord)
        .where(
            ArtifactRecord.job_id == job.id,
            ArtifactRecord.attempt == job.attempt,
            ArtifactRecord.available.is_(True),
        )
        .order_by(ArtifactRecord.kind, ArtifactRecord.revision.desc())
    ).all()
    newest: dict[tuple[str, str], ArtifactRecord] = {}
    for row in rows:
        newest.setdefault((row.kind, str(row.declared.get("media_type"))), row)
    return [to_manifest(r) for r in newest.values()]


def sampling_info(job: JobRecord) -> SamplingInfo | None:
    if job.frame_count is None or job.fps_effective is None:
        return None
    return SamplingInfo(
        auto=job.fps_auto,
        fps_requested=job.fps_effective if job.fps_auto else job.fps,
        fps_effective=job.fps_effective,
        frames=job.frame_count,
        cap=MAX_FRAMES[ComputeTarget(job.compute)],
        clamped=job.frames_clamped,
    )


def snapshot(db: Session, job: JobRecord) -> Job:
    stages = {s: StageState(stage=s, state=StageRunState.PENDING) for s in PIPELINE_STAGES}
    warnings: list[WarningData] = []
    error: str | None = None
    retryable = False
    estimate: JobEstimateData | None = None
    running: StageName | None = None
    for payload in db.scalars(
        select(JobEventRecord.payload)
        .where(JobEventRecord.job_id == job.id)
        .order_by(JobEventRecord.seq)
    ):
        type_ = EventType(payload["type"])
        stage = StageName(payload["stage"]) if payload.get("stage") else None
        data = payload.get("data") or {}
        if type_ is EventType.JOB_QUEUED:  # a new attempt starts from a clean slate
            stages = {s: StageState(stage=s, state=StageRunState.PENDING) for s in PIPELINE_STAGES}
            warnings, error, retryable, running = [], None, False, None
            estimate = None
        elif type_ is EventType.STAGE_STARTED and stage:
            stages[stage] = StageState(
                stage=stage, state=StageRunState.RUNNING, started_at=payload["ts"]
            )
            running = stage
        elif type_ is EventType.STAGE_PROGRESS and stage:
            stages[stage] = stages[stage].model_copy(
                update={"progress": StageProgressData.model_validate(data)}
            )
        elif type_ is EventType.STAGE_COMPLETED and stage:
            stages[stage] = StageState(
                stage=stage,
                state=StageRunState.COMPLETED,
                duration_s=data.get("duration_s"),
                peak_vram_mib=data.get("peak_vram_mib"),
                budget_mib=data.get("budget_mib"),
                started_at=stages[stage].started_at,
            )
            running = None
        elif type_ is EventType.JOB_ESTIMATE:
            estimate = JobEstimateData.model_validate(data)
        elif type_ is EventType.STAGE_SKIPPED and stage:
            stages[stage] = StageState(
                stage=stage, state=StageRunState.SKIPPED, reason=data.get("reason")
            )
        elif type_ is EventType.WARNING:
            warnings.append(WarningData.model_validate(data))
        elif type_ is EventType.JOB_FAILED:
            error, retryable = data.get("reason"), bool(data.get("retryable"))
            if running:
                stages[running] = StageState(
                    stage=running, state=StageRunState.FAILED, reason=error
                )
        elif type_ is EventType.JOB_CANCELLED and running:
            stages[running] = StageState(stage=running, state=StageRunState.CANCELLED)
    return Job(
        id=job.id,
        mission_id=job.mission_id,
        status=JobStatus(job.status),
        pipeline=PipelineMode(job.pipeline),
        compute=ComputeTarget(job.compute),
        attempt=job.attempt,
        last_seq=job.last_seq,
        stages=list(stages.values()),
        artifacts=current_artifacts(db, job),
        warnings=warnings,
        error=error,
        retryable=retryable,
        sampling=sampling_info(job),
        estimate=estimate,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def requeue(db: Session, job: JobRecord, reason: str) -> None:
    job.attempt += 1
    job.cancel_requested = False
    job.lease_id = job.lease_worker = job.lease_expires_at = None
    append_event(db, job, EventType.JOB_QUEUED, data={"attempt": job.attempt, "reason": reason})


def request_cancel(db: Session, job: JobRecord) -> None:
    """Cancel immediately when no worker holds the job; otherwise ask the worker to stop."""
    status = JobStatus(job.status)
    if status in TERMINAL_STATUSES or status is JobStatus.CANCEL_REQUESTED:
        raise EventRejected("This job has already finished or is being cancelled.")
    job.cancel_requested = True
    append_event(db, job, EventType.CANCEL_REQUESTED)
    if job.lease_id is None:
        append_event(db, job, EventType.JOB_CANCELLED)


def expire_leases(db: Session) -> int:
    """Return jobs whose worker stopped renewing its lease to the queue (or finish their
    cancellation). Safe to call from any request or a periodic sweep."""
    now = utcnow()
    stale = db.scalars(
        select(JobRecord).where(
            JobRecord.lease_expires_at.is_not(None), JobRecord.lease_expires_at < now
        )
    ).all()
    for job in stale:
        status = JobStatus(job.status)
        if job.cancel_requested:
            job.lease_id = None
            append_event(db, job, EventType.JOB_CANCELLED)
        elif status is JobStatus.QUEUED:  # leased but never started
            job.lease_id = job.lease_worker = job.lease_expires_at = None
        else:
            requeue(db, job, "The worker stopped responding; the job went back to the queue.")
    db.commit()
    return len(stale)


@dataclass(frozen=True)
class Lease:
    job: JobRecord
    lease_id: str


def lease_next(
    db: Session,
    worker_id: str,
    modes: list[PipelineMode],
    seconds: int,
    job_id: str | None = None,
    location: ComputeTarget = ComputeTarget.LOCAL,
) -> Lease | None:
    """Lease the oldest matching queued job aimed at `location`, or only `job_id` when given
    (bundle import)."""
    expire_leases(db)
    query = select(JobRecord).where(
        JobRecord.status == JobStatus.QUEUED,
        JobRecord.compute == location.value,
        JobRecord.lease_id.is_(None),
        JobRecord.cancel_requested.is_(False),
        JobRecord.pipeline.in_([m.value for m in modes]),
    )
    if job_id is not None:
        query = query.where(JobRecord.id == job_id)
    candidates = db.scalars(query.order_by(JobRecord.created_at).limit(5)).all()
    for job in candidates:
        lease_id = secrets.token_urlsafe(24)
        result = db.execute(
            update(JobRecord)
            .where(JobRecord.id == job.id, JobRecord.lease_id.is_(None))
            .values(
                lease_id=lease_id,
                lease_worker=worker_id,
                lease_expires_at=utcnow() + timedelta(seconds=seconds),
            )
        )
        db.commit()
        if getattr(result, "rowcount", 0):  # another worker may have won the race for this row
            db.refresh(job)
            return Lease(job, lease_id)
    return None


def next_revision(db: Session, job_id: str, kind: str) -> int:
    top = db.scalar(
        select(func.max(ArtifactRecord.revision)).where(
            ArtifactRecord.job_id == job_id, ArtifactRecord.kind == kind
        )
    )
    return (top or 0) + 1
