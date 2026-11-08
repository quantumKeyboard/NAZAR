"""Health (liveness) and readiness contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from nazar_contracts.job_api import GeometryBackend, ModelReadiness, TimeModel
from nazar_contracts.jobs import ComputeTarget


class CheckStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    NOT_IMPLEMENTED = "not_implemented"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    time: datetime


class ReadinessCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    required: bool


class WorkerPresence(BaseModel):
    """Whether a worker at this location polled the API recently (presence, not a secret)."""

    location: ComputeTarget
    online: bool
    last_seen_s: float | None = Field(description="Seconds since its last call; None: never.")
    geometry: GeometryBackend | None = Field(
        description="Geometry backend it reported for standard jobs (None: not reported)."
    )
    models: list[ModelReadiness] | None = Field(
        default=None, description="Models it can run right now (None: not reported)."
    )
    time_model: TimeModel | None = Field(
        default=None, description="Its time per frame, for estimates before a job (None: unknown)."
    )


class ReadinessResponse(BaseModel):
    ready: bool
    checks: list[ReadinessCheck]
    workers: list[WorkerPresence] = Field(
        default_factory=list, description="One entry per compute location."
    )
