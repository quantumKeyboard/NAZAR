"""Readiness checks. Each check reports presence/connectivity only - never a secret value."""

import logging
import uuid
from datetime import UTC, datetime

from pydantic import SecretStr
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from nazar_api.db import get_engine
from nazar_api.models import User
from nazar_api.settings import Settings
from nazar_contracts import (
    CheckStatus,
    ComputeTarget,
    GeometryBackend,
    ModelReadiness,
    ReadinessCheck,
    ReadinessResponse,
    TimeModel,
    WorkerPresence,
)

log = logging.getLogger(__name__)


def _database(settings: Settings) -> ReadinessCheck:
    kind = "PostgreSQL" if settings.database_url else "SQLite (local default)"
    try:
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        with get_engine(settings.sqlalchemy_url).connect() as conn:
            conn.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError) as exc:
        log.warning("database check failed", extra={"error": type(exc).__name__})
        return ReadinessCheck(
            name="database",
            status=CheckStatus.UNAVAILABLE,
            detail=f"{kind} is unreachable ({type(exc).__name__}).",
            required=True,
        )
    return ReadinessCheck(
        name="database", status=CheckStatus.OK, detail=f"{kind} reachable.", required=True
    )


def _storage(settings: Settings) -> ReadinessCheck:
    if settings.s3_endpoint_url:
        return ReadinessCheck(
            name="storage",
            status=CheckStatus.NOT_IMPLEMENTED,
            detail="S3-compatible storage is configured but its adapter is not implemented yet.",
            required=True,
        )
    probe = settings.storage_path / f".probe-{uuid.uuid4().hex}"
    try:
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        return ReadinessCheck(
            name="storage",
            status=CheckStatus.UNAVAILABLE,
            detail=f"Local storage directory is not writable ({type(exc).__name__}).",
            required=True,
        )
    return ReadinessCheck(
        name="storage", status=CheckStatus.OK, detail="Local storage writable.", required=True
    )


def _secret(name: str, value: SecretStr | str | None, purpose: str) -> ReadinessCheck:
    present = value is not None and (
        value.get_secret_value() if isinstance(value, SecretStr) else value
    )
    return ReadinessCheck(
        name=f"config:{name}",
        status=CheckStatus.OK if present else CheckStatus.NOT_CONFIGURED,
        detail=f"{purpose}. {'Present.' if present else 'Blank in .env.'}",
        required=True,
    )


def _accounts(settings: Settings) -> ReadinessCheck:
    try:
        with get_engine(settings.sqlalchemy_url).connect() as conn:
            users = conn.execute(select(func.count()).select_from(User)).scalar_one()
    except SQLAlchemyError as exc:
        return ReadinessCheck(
            name="accounts",
            status=CheckStatus.UNAVAILABLE,
            detail=f"Accounts table is unreadable ({type(exc).__name__}).",
            required=True,
        )
    if users:
        return ReadinessCheck(
            name="accounts",
            status=CheckStatus.OK,
            detail="Operator account exists.",
            required=True,
        )
    return ReadinessCheck(
        name="accounts",
        status=CheckStatus.NOT_CONFIGURED,
        detail="No operator account yet. Create one in the web app (first-run setup).",
        required=True,
    )


WORKER_FRESH_S = 60
LOCATION_TEXT = {ComputeTarget.LOCAL: "local", ComputeTarget.COLLEGE: "college"}

Seen = dict[ComputeTarget, datetime]
Geometry = dict[ComputeTarget, GeometryBackend]
Models = dict[ComputeTarget, list[ModelReadiness]]
TimeModels = dict[ComputeTarget, TimeModel]


def _age(seen: Seen, location: ComputeTarget) -> float | None:
    at = seen.get(location)
    return None if at is None else (datetime.now(UTC) - at).total_seconds()


def _fresh(seen: Seen) -> list[ComputeTarget]:
    return [
        loc
        for loc in ComputeTarget
        if (age := _age(seen, loc)) is not None and age <= WORKER_FRESH_S
    ]


def _worker(seen: Seen) -> ReadinessCheck:
    fresh = _fresh(seen)
    if fresh:
        parts = [f"{LOCATION_TEXT[loc]} {_age(seen, loc):.0f} s ago" for loc in fresh]
        return ReadinessCheck(
            name="worker",
            status=CheckStatus.OK,
            detail=f"Worker contact: {', '.join(parts)}. Jobs run only on the location chosen "
            "when they were started.",
            required=False,
        )
    return ReadinessCheck(
        name="worker",
        status=CheckStatus.NOT_CONFIGURED,
        detail="No worker has contacted the API in the last minute. Jobs wait in the queue "
        "until one does (locally: `pnpm worker`).",
        required=False,
    )


def _gpu(seen: Seen, geometry: Geometry) -> ReadinessCheck:
    fresh = [loc for loc in _fresh(seen) if loc in geometry]
    fast3r = [LOCATION_TEXT[loc] for loc in fresh if geometry[loc] is GeometryBackend.FAST3R]
    if fast3r:
        return ReadinessCheck(
            name="gpu",
            status=CheckStatus.OK,
            detail=f"Fast3R GPU runtime on the {' and '.join(fast3r)} worker: Reconstruct there "
            "uses dense learned geometry. Each job report records the GPU and memory it used.",
            required=False,
        )
    if fresh:
        return ReadinessCheck(
            name="gpu",
            status=CheckStatus.NOT_CONFIGURED,
            detail="The running worker has no Fast3R GPU runtime, so Reconstruct uses the CPU "
            "baseline (sparse). Set it up with `pnpm gpu:setup` (docs/ENVIRONMENT_SETUP.md).",
            required=False,
        )
    return ReadinessCheck(
        name="gpu",
        status=CheckStatus.NOT_CONFIGURED,
        detail="No worker running standard jobs has reported its GPU runtime in the last minute.",
        required=False,
    )


def _presence(
    seen: Seen,
    geometry: Geometry,
    models: Models | None = None,
    time_models: TimeModels | None = None,
) -> list[WorkerPresence]:
    fresh = _fresh(seen)
    return [
        WorkerPresence(
            location=loc,
            online=loc in fresh,
            last_seen_s=None if (age := _age(seen, loc)) is None else round(age, 1),
            geometry=geometry.get(loc),
            models=(models or {}).get(loc),
            time_model=(time_models or {}).get(loc),
        )
        for loc in ComputeTarget
    ]


def check_readiness(
    settings: Settings,
    worker_seen_at: Seen | None = None,
    worker_geometry: Geometry | None = None,
    worker_models: Models | None = None,
    worker_time_model: TimeModels | None = None,
) -> ReadinessResponse:
    seen, geometry = worker_seen_at or {}, worker_geometry or {}
    checks = [
        _database(settings),
        _storage(settings),
        _secret("NAZAR_SECRET_KEY", settings.secret_key, "Signs artifact links and sessions"),
        _secret("NAZAR_WORKER_TOKEN", settings.worker_token, "Authenticates the GPU worker"),
        _accounts(settings),
        _worker(seen),
        _gpu(seen, geometry),
    ]
    ready = all(c.status is CheckStatus.OK for c in checks if c.required)
    return ReadinessResponse(
        ready=ready,
        checks=checks,
        workers=_presence(seen, geometry, worker_models, worker_time_model),
    )
