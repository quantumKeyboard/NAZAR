"""Persistent records. Portable between SQLite (local default) and PostgreSQL."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from nazar_api.db import Base, UTCDateTime


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class AuthSession(Base):
    """Server-side session. Only a SHA-256 of the cookie token is stored."""

    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())


class MissionRecord(Base):
    __tablename__ = "missions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class AssetRecord(Base):
    """An uploaded input file. `stored_name` is generated; the user's filename is display-only."""

    __tablename__ = "assets"
    __table_args__ = (Index("ix_assets_mission_kind", "mission_id", "kind"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))
    filename: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(BigInteger)
    received_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    sha256: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class JobRecord(Base):
    """A reconstruction job. `status` and `last_seq` are a cache of the event log."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    mission_id: Mapped[str] = mapped_column(
        ForeignKey("missions.id", ondelete="CASCADE"), index=True
    )
    video_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    # Pinned at creation. No FK: removing the telemetry must not delete or block the job;
    # the worker is told it is gone and records that.
    telemetry_asset_id: Mapped[str | None] = mapped_column(String(32), default=None)
    pipeline: Mapped[str] = mapped_column(String(16))
    compute: Mapped[str] = mapped_column(String(16), default="local", server_default="local")
    # Frame sampling (standard pipeline): the rate asked for and what the video gave.
    fps: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")
    fps_auto: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    frame_count: Mapped[int | None] = mapped_column(Integer, default=None)
    fps_effective: Mapped[float | None] = mapped_column(Float, default=None)
    frames_clamped: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    status: Mapped[str] = mapped_column(String(24), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    last_seq: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    lease_id: Mapped[str | None] = mapped_column(String(64), default=None)
    lease_worker: Mapped[str | None] = mapped_column(String(64), default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class JobEventRecord(Base):
    """Append-only event log. `(job_id, seq)` is the client cursor; `key` deduplicates."""

    __tablename__ = "job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "seq", name="uq_job_events_seq"),
        UniqueConstraint("job_id", "key", name="uq_job_events_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    key: Mapped[str | None] = mapped_column(String(96), default=None)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ArtifactRecord(Base):
    """A job output. Bytes live at a server-generated path; `manifest` is the public view."""

    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("job_id", "key", name="uq_artifacts_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(96))
    kind: Mapped[str] = mapped_column(String(32))
    revision: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer)
    final: Mapped[bool] = mapped_column(Boolean, default=False)
    available: Mapped[bool] = mapped_column(Boolean, default=False)
    declared: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
