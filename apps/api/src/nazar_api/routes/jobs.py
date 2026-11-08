"""Browser-facing job API: start, snapshot, event replay, live stream, cancel, retry,
and authorised artifact downloads."""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from nazar_api.deps import DbDep, SettingsDep, UserDep
from nazar_api.jobs import (
    EventRejected,
    append_event,
    events_after,
    request_cancel,
    requeue,
    snapshot,
)
from nazar_api.models import ArtifactRecord, AssetRecord, JobRecord, MissionRecord, User
from nazar_api.routes.missions import _owned_mission
from nazar_api.security import SESSION_COOKIE, user_for_token
from nazar_api.settings import Settings
from nazar_api.storage import PREFLIGHT_DIR, artifact_path, asset_dir, job_frames_dir, remove_tree
from nazar_contracts import (
    AUTO_MAX_FRAMES,
    FPS_DEFAULT,
    MAX_FRAMES,
    TERMINAL_STATUSES,
    AssetKind,
    AssetStatus,
    EventType,
    Job,
    JobCreate,
    JobStatus,
    PipelineMode,
    PreflightReport,
)
from nazar_media import MediaError
from nazar_media.sampling import sample_job_frames

router = APIRouter(prefix="/api", tags=["jobs"])

STREAM_BATCH = 200
STREAM_IDLE_S = 0.25


def owned_job(db: Session, user: User, job_id: str) -> JobRecord:
    job = db.execute(
        select(JobRecord)
        .join(MissionRecord, MissionRecord.id == JobRecord.mission_id)
        .where(JobRecord.id == job_id, MissionRecord.owner_id == user.id)
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return job


@router.post("/missions/{mission_id}/jobs", response_model=Job, status_code=201)
def create_job(
    mission_id: str, body: JobCreate, user: UserDep, db: DbDep, settings: SettingsDep
) -> Job:
    """Queue reconstruction of the mission's validated video. For the standard pipeline the
    frames are sampled first (at `body.fps`, or automatically), so a video that cannot give
    enough frames is refused here with a reason instead of failing later on the GPU."""
    mission = _owned_mission(db, user, mission_id)
    video = db.scalars(
        select(AssetRecord).where(
            AssetRecord.mission_id == mission.id,
            AssetRecord.kind == AssetKind.VIDEO,
            AssetRecord.status == AssetStatus.READY,
        )
    ).first()
    if video is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Upload a video and wait for validation to finish first."
        )
    active = db.scalars(
        select(JobRecord.id).where(
            JobRecord.mission_id == mission.id,
            JobRecord.status.not_in([s.value for s in TERMINAL_STATUSES]),
        )
    ).first()
    if active is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This mission already has a running job.")
    telemetry = db.scalars(
        select(AssetRecord.id).where(
            AssetRecord.mission_id == mission.id,
            AssetRecord.kind == AssetKind.TELEMETRY,
            AssetRecord.status == AssetStatus.READY,
        )
    ).first()
    job = JobRecord(
        mission_id=mission.id,
        video_asset_id=video.id,
        telemetry_asset_id=telemetry,
        pipeline=body.pipeline.value,
        compute=body.compute.value,
        fps=body.fps if body.fps is not None else FPS_DEFAULT,
        fps_auto=body.fps is None,
        status=JobStatus.CREATED.value,
        attempt=1,
        last_seq=0,
    )
    db.add(job)
    db.flush()
    if body.pipeline is PipelineMode.STANDARD:
        _sample_frames(db, settings, job, video, body)
    append_event(
        db,
        job,
        EventType.JOB_CREATED,
        data={"mission_id": mission.id, "test_pipeline": body.pipeline == "test"},
    )
    append_event(
        db, job, EventType.JOB_QUEUED, data={"attempt": 1, "reason": "Waiting for a worker."}
    )
    db.commit()
    return snapshot(db, job)


def _sample_frames(
    db: Session, settings: Settings, job: JobRecord, video: AssetRecord, body: JobCreate
) -> None:
    folder = asset_dir(settings, video.mission_id, video.id)
    report_file = folder / PREFLIGHT_DIR / "preflight.json"
    cap = MAX_FRAMES[body.compute]
    try:
        report = PreflightReport.model_validate_json(report_file.read_text(encoding="utf-8"))
        plan = sample_job_frames(
            folder / video.stored_name,
            report,
            body.fps,
            cap,
            job_frames_dir(settings, job.id),
            auto_max=AUTO_MAX_FRAMES[body.compute],
        )
    except MediaError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.message) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        db.rollback()
        remove_tree(job_frames_dir(settings, job.id))
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"The video's frames could not be sampled ({exc}).",
        ) from exc
    job.frame_count = len(plan.keyframes)
    job.fps_effective = plan.fps_effective
    job.frames_clamped = plan.clamped


@router.get("/missions/{mission_id}/jobs", response_model=list[Job])
def list_jobs(mission_id: str, user: UserDep, db: DbDep) -> list[Job]:
    mission = _owned_mission(db, user, mission_id)
    rows = db.scalars(
        select(JobRecord)
        .where(JobRecord.mission_id == mission.id)
        .order_by(JobRecord.created_at.desc())
        .limit(20)
    ).all()
    return [snapshot(db, r) for r in rows]


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str, user: UserDep, db: DbDep) -> Job:
    return snapshot(db, owned_job(db, user, job_id))


@router.get("/jobs/{job_id}/events")
def list_events(
    job_id: str,
    user: UserDep,
    db: DbDep,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=STREAM_BATCH, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """REST replay / polling fallback: events with `seq > after`, oldest first."""
    job = owned_job(db, user, job_id)
    return events_after(db, job.id, after, limit)


@router.post("/jobs/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str, user: UserDep, db: DbDep) -> Job:
    job = owned_job(db, user, job_id)
    try:
        request_cancel(db, job)
    except EventRejected as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    db.commit()
    return snapshot(db, job)


@router.post("/jobs/{job_id}/retry", response_model=Job)
def retry_job(job_id: str, user: UserDep, db: DbDep) -> Job:
    job = owned_job(db, user, job_id)
    current = snapshot(db, job)
    allowed = current.status is JobStatus.CANCELLED or (
        current.status is JobStatus.FAILED and current.retryable
    )
    if not allowed:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Only a cancelled or retryable failed job can be retried."
        )
    requeue(db, job, "Retry requested.")
    db.commit()
    return snapshot(db, job)


@router.get("/artifacts/{artifact_id}/content")
def get_artifact(artifact_id: str, user: UserDep, db: DbDep, settings: SettingsDep) -> FileResponse:
    row = db.get(ArtifactRecord, artifact_id)
    if row is None or not row.available:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found.")
    owned_job(db, user, row.job_id)
    path = artifact_path(settings, row.job_id, row.id)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found.")
    # Content is immutable per id: a new revision is a new artifact id.
    return FileResponse(
        path,
        media_type=row.declared["media_type"],
        headers={
            "Cache-Control": "private, max-age=86400, immutable",
            "ETag": f'"{row.declared["checksum_sha256"]}"',
        },
    )


# --- Live stream ---------------------------------------------------------------------------


def _origin_allowed(ws: WebSocket, settings: Settings) -> bool:
    """Browsers always send Origin on WebSocket handshakes; reject foreign pages so a
    session cookie cannot be used by another site (cross-site WebSocket hijacking)."""
    origin = ws.headers.get("origin")
    if origin is None:
        return True
    host = ws.headers.get("host", "")
    return origin in settings.web_origins or origin in (f"http://{host}", f"https://{host}")


def _authorise(ws: WebSocket, job_id: str) -> str | None:
    with Session(ws.app.state.engine) as db:
        user = user_for_token(db, ws.cookies.get(SESSION_COOKIE))
        if user is None:
            return None
        try:
            return owned_job(db, user, job_id).id
        except HTTPException:
            return None


def _read(ws: WebSocket, job_id: str, after: int) -> tuple[list[dict[str, Any]], bool]:
    with Session(ws.app.state.engine) as db:
        job = db.get(JobRecord, job_id)
        if job is None:
            return [], True
        events = events_after(db, job_id, after, STREAM_BATCH)
        return events, JobStatus(job.status) in TERMINAL_STATUSES


@router.websocket("/jobs/{job_id}/stream")
async def stream_events(ws: WebSocket, job_id: str, after: int = Query(default=0, ge=0)) -> None:
    """Push events with `seq > after` as they are appended. The connection closes normally
    once a terminal event has been delivered. Clients reconnect with the last seq they
    applied; nothing is lost because the log is persistent."""
    if not _origin_allowed(ws, ws.app.state.settings):
        await ws.close(code=4403)
        return
    if await run_in_threadpool(_authorise, ws, job_id) is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    # Watch for the client going away while we idle between polls.
    receiver = asyncio.ensure_future(ws.receive())
    cursor = after
    try:
        while True:
            events, terminal = await run_in_threadpool(_read, ws, job_id, cursor)
            for event in events:
                # send_json awaits the transport, so a slow client applies backpressure
                # and we never buffer more than one batch.
                await ws.send_json(event)
                cursor = event["seq"]
            if not events and terminal:
                await ws.close(code=1000)
                return
            if not events:
                done, _ = await asyncio.wait({receiver}, timeout=STREAM_IDLE_S)
                if done:
                    message = receiver.result()
                    if message["type"] == "websocket.disconnect":
                        return
                    receiver = asyncio.ensure_future(ws.receive())  # ignore client chatter
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        receiver.cancel()
