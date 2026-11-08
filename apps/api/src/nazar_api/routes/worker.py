"""Worker protocol (ADR-0003): the worker makes only outbound calls to these routes.

Every call carries `Authorization: Bearer <NAZAR_WORKER_TOKEN>`. Calls about a job also
carry the `X-Nazar-Lease` id handed out by `/lease`; a worker whose lease lapsed (and whose
job went back to the queue) gets 409 and must drop the job.
"""

import hmac
import os
from datetime import timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from nazar_api.deps import DbDep, SettingsDep
from nazar_api.jobs import (
    WORKER_EVENT_TYPES,
    EventRejected,
    append_event,
    expire_leases,
    lease_next,
    next_revision,
    sampling_info,
    to_manifest,
)
from nazar_api.models import ArtifactRecord, AssetRecord, JobRecord, utcnow
from nazar_api.settings import Settings
from nazar_api.storage import (
    PREFLIGHT_DIR,
    artifact_path,
    asset_dir,
    job_frames_dir,
    sha256_file,
)
from nazar_contracts import (
    TERMINAL_STATUSES,
    ArtifactCreate,
    ArtifactCreated,
    ArtifactKind,
    AssetStatus,
    ComputeTarget,
    EventType,
    JobStatus,
    PipelineMode,
    StageName,
    WorkerEventAck,
    WorkerEventBatch,
    WorkerHeartbeat,
    WorkerLease,
    WorkerLeaseRequest,
)

PREVIEW_KINDS = frozenset({ArtifactKind.POINT_CLOUD_PREVIEW, ArtifactKind.POINT_CLOUD})


def require_worker(
    request: Request,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
    x_nazar_worker_location: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.worker_token
    if expected is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "NAZAR_WORKER_TOKEN is not configured."
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        token.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid worker token.")
    try:
        location = ComputeTarget(x_nazar_worker_location or ComputeTarget.LOCAL.value)
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "X-Nazar-Worker-Location must be local or college."
        ) from None
    request.app.state.worker_seen_at[location] = utcnow()


router = APIRouter(prefix="/api/worker", tags=["worker"], dependencies=[Depends(require_worker)])


def leased_job(
    job_id: str, db: DbDep, x_nazar_lease: Annotated[str | None, Header()] = None
) -> JobRecord:
    expire_leases(db)
    job = db.get(JobRecord, job_id)
    if (
        job is None
        or job.lease_id is None
        or x_nazar_lease is None
        or not hmac.compare_digest(job.lease_id.encode(), x_nazar_lease.encode())
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "Lease lost; drop this job.")
    return job


LeasedJob = Annotated[JobRecord, Depends(leased_job)]


@router.post("/lease", response_model=WorkerLease, responses={204: {"description": "No work"}})
def lease(
    body: WorkerLeaseRequest, request: Request, db: DbDep, settings: SettingsDep
) -> WorkerLease | Response:
    if body.geometry is not None:
        request.app.state.worker_geometry[body.location] = body.geometry
    if body.models is not None:
        request.app.state.worker_models[body.location] = body.models
    if body.time_model is not None:
        request.app.state.worker_time_model[body.location] = body.time_model
    got = lease_next(
        db, body.worker_id, body.modes, settings.worker_lease_seconds, body.job_id, body.location
    )
    if got is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    job = got.job
    video = db.get(AssetRecord, job.video_asset_id)
    if video is None or video.sha256 is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "The job's video is no longer available.")
    keyframes = _frames_root(settings, db, job) / "keyframes"
    count = len(list(keyframes.glob("*.jpg"))) if keyframes.is_dir() else 0
    assert job.lease_expires_at is not None
    return WorkerLease(
        lease_id=got.lease_id,
        job_id=job.id,
        mission_id=job.mission_id,
        pipeline=PipelineMode(job.pipeline),
        attempt=job.attempt,
        lease_expires_at=job.lease_expires_at,
        video_asset_id=video.id,
        keyframe_count=count,
        sampling=sampling_info(job),
        video_sha256=video.sha256,
        telemetry_asset_id=job.telemetry_asset_id,
    )


@router.post("/jobs/{job_id}/heartbeat", response_model=WorkerHeartbeat)
def heartbeat(job: LeasedJob, db: DbDep, settings: SettingsDep) -> WorkerHeartbeat:
    job.lease_expires_at = utcnow() + timedelta(seconds=settings.worker_lease_seconds)
    db.commit()
    return WorkerHeartbeat(
        cancel_requested=job.cancel_requested, lease_expires_at=job.lease_expires_at
    )


def _completion_data(db: Session, job: JobRecord) -> dict[str, list[str]]:
    rows = db.scalars(
        select(ArtifactRecord).where(
            ArtifactRecord.job_id == job.id, ArtifactRecord.attempt == job.attempt
        )
    ).all()
    if any(not r.available for r in rows):
        raise EventRejected("Some declared artifacts were never uploaded.")
    final = [r.id for r in rows if r.final]
    if not final:
        raise EventRejected("A job cannot complete without a validated final artifact.")
    return {"artifact_ids": final}


@router.post("/jobs/{job_id}/events", response_model=WorkerEventAck)
def post_events(body: WorkerEventBatch, job: LeasedJob, db: DbDep) -> WorkerEventAck:
    """Append events in order. Each is committed on its own and deduplicated by key, so a
    worker that lost the response can resend the whole batch safely."""
    for event in body.events:
        if event.type not in WORKER_EVENT_TYPES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Workers cannot send {event.type.value} events.",
            )
        try:
            data = event.data
            if event.type is EventType.JOB_COMPLETED:
                data = _completion_data(db, job)
            append_event(
                db,
                job,
                event.type,
                stage=event.stage,
                data=data,
                key=f"{job.attempt}:{event.key}",
            )
            db.commit()
        except EventRejected as exc:
            db.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, f"{event.key}: {exc}") from exc
    return WorkerEventAck(
        status=JobStatus(job.status), last_seq=job.last_seq, cancel_requested=job.cancel_requested
    )


@router.get("/jobs/{job_id}/inputs/preflight")
def input_preflight(job: LeasedJob, db: DbDep, settings: SettingsDep) -> FileResponse:
    return FileResponse(_frames_root(settings, db, job) / "preflight.json")


@router.get("/jobs/{job_id}/inputs/keyframes/{number}.jpg")
def input_keyframe(number: int, job: LeasedJob, db: DbDep, settings: SettingsDep) -> FileResponse:
    path = _frames_root(settings, db, job) / "keyframes" / f"{number:04d}.jpg"
    if not 1 <= number <= 9999 or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Keyframe not found.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/jobs/{job_id}/inputs/telemetry")
def input_telemetry(job: LeasedJob, db: DbDep, settings: SettingsDep) -> FileResponse:
    """The normalised track (nazar.telemetry-track/1) validated at upload."""
    row = db.get(AssetRecord, job.telemetry_asset_id) if job.telemetry_asset_id else None
    path = asset_dir(settings, row.mission_id, row.id) / "track.json" if row else None
    if row is None or row.status != AssetStatus.READY or path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This job's telemetry was removed.")
    return FileResponse(path, media_type="application/json")


@router.get("/jobs/{job_id}/inputs/video")
def input_video(job: LeasedJob, db: DbDep, settings: SettingsDep) -> FileResponse:
    video = db.get(AssetRecord, job.video_asset_id)
    assert video is not None
    return FileResponse(_input_dir(db, settings, job) / video.stored_name)


def _frames_root(settings: Settings, db: Session, job: JobRecord) -> Path:
    """The job's own sampled frames when it has them, else the upload's preflight keyframes
    (test pipeline, and jobs created before frame sampling existed)."""
    own = job_frames_dir(settings, job.id)
    if (own / "preflight.json").is_file():
        return own
    return _input_dir(db, settings, job) / PREFLIGHT_DIR


def _input_dir(db: Session, settings: SettingsDep, job: JobRecord) -> Path:
    video = db.get(AssetRecord, job.video_asset_id)
    if video is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Input not found.")
    return asset_dir(settings, video.mission_id, video.id)


@router.post("/jobs/{job_id}/artifacts", response_model=ArtifactCreated)
def declare_artifact(
    body: ArtifactCreate, job: LeasedJob, db: DbDep, settings: SettingsDep
) -> ArtifactCreated:
    """Declare an artifact before uploading its bytes. Idempotent by `key`."""
    if JobStatus(job.status) in TERMINAL_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "The job has finished.")
    if body.byte_size > settings.max_artifact_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Artifact exceeds the size limit.")
    key = f"{job.attempt}:{body.key}"
    existing = db.scalars(
        select(ArtifactRecord).where(ArtifactRecord.job_id == job.id, ArtifactRecord.key == key)
    ).first()
    if existing is not None:
        return ArtifactCreated(
            id=existing.id, revision=existing.revision, uploaded=existing.available
        )
    row = ArtifactRecord(
        job_id=job.id,
        key=key,
        kind=body.kind.value,
        revision=next_revision(db, job.id, body.kind.value),
        attempt=job.attempt,
        final=body.final,
        available=False,
        declared=body.model_dump(mode="json"),
    )
    db.add(row)
    db.commit()
    return ArtifactCreated(id=row.id, revision=row.revision, uploaded=False)


@router.put("/jobs/{job_id}/artifacts/{artifact_id}/content", response_model=ArtifactCreated)
async def upload_artifact(
    artifact_id: str, request: Request, job: LeasedJob, db: DbDep, settings: SettingsDep
) -> ArtifactCreated:
    """Receive the whole artifact body, verify size and SHA-256, publish it atomically, then
    announce it. A failed verification leaves nothing visible; the worker may resend."""
    row = db.get(ArtifactRecord, artifact_id)
    if row is None or row.job_id != job.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found.")
    if row.available:
        return ArtifactCreated(id=row.id, revision=row.revision, uploaded=True)
    size = int(row.declared["byte_size"])
    path = artifact_path(settings, job.id, row.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    written = 0
    with part.open("wb") as fh:
        async for piece in request.stream():
            written += len(piece)
            if written > size:
                break
            fh.write(piece)
    if written != size or sha256_file(part) != row.declared["checksum_sha256"]:
        part.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Artifact bytes do not match the declared size and checksum.",
        )
    os.replace(part, path)
    row.available = True
    try:
        append_event(
            db,
            job,
            EventType.ARTIFACT_AVAILABLE,
            stage=_running_stage(job),
            data=_manifest_json(row),
            key=f"{job.attempt}:artifact:{row.id}",
        )
        if ArtifactKind(row.kind) in PREVIEW_KINDS:
            append_event(
                db,
                job,
                EventType.PREVIEW_UPDATED,
                stage=_running_stage(job),
                data={"artifact_id": row.id, "revision": row.revision},
                key=f"{job.attempt}:preview:{row.id}",
            )
    except EventRejected as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    db.commit()
    return ArtifactCreated(id=row.id, revision=row.revision, uploaded=True)


def _running_stage(job: JobRecord) -> StageName | None:
    try:
        return StageName(job.status)
    except ValueError:
        return None


def _manifest_json(row: ArtifactRecord) -> dict[str, object]:
    return to_manifest(row).model_dump(mode="json")
