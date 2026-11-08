"""Video probe and preflight (frame quality + keyframe selection) contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

PREFLIGHT_SCHEMA_VERSION: Literal[1] = 1


class VideoProbe(BaseModel):
    """Facts read from the container/stream by ffprobe. `display_*` include rotation."""

    container: str
    codec: str
    pix_fmt: str | None
    width: int
    height: int
    display_width: int
    display_height: int
    rotation_deg: int
    duration_s: float
    frame_count: int | None
    nominal_fps: float | None
    average_fps: float | None
    variable_frame_rate: bool
    bit_rate: int | None
    has_audio: bool


class FrameFlag(StrEnum):
    BLURRY = "blurry"
    DARK = "dark"
    OVEREXPOSED = "overexposed"
    DUPLICATE = "duplicate"


class FrameSeries(BaseModel):
    """Per-frame metrics, column-oriented (one entry per decoded frame, in display order).

    - `t`: presentation time in seconds from the stream's first frame.
    - `sharpness`: variance of the Laplacian at analysis resolution (higher = sharper).
    - `brightness`: mean luma, 0-1.
    - `change`: mean absolute luma difference to the previous frame, 0-1.
    - `motion_px`: estimated global image shift to the previous frame, in analysis pixels.
    - `flags`: per-frame quality flags.
    """

    index: list[int]
    t: list[float]
    sharpness: list[float]
    brightness: list[float]
    change: list[float]
    motion_px: list[float]
    flags: list[list[FrameFlag]]


class KeyframeReason(StrEnum):
    START = "start"
    NOVELTY = "novelty"
    MAX_GAP = "max_gap"
    END = "end"
    SAMPLED = "sampled"


class Keyframe(BaseModel):
    index: int
    t: float
    reason: KeyframeReason
    quality: float
    image: int
    """1-based number of the extracted image (`/keyframes/{image}`)."""


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class PreflightWarning(BaseModel):
    code: str
    severity: WarningSeverity
    message: str


class AnalysisInfo(BaseModel):
    width: int
    height: int
    frames_decoded: int
    decode_errors: int
    timestamps: Literal["stream", "estimated"]
    processing_s: float


class PreflightSummary(BaseModel):
    usable_frames: int
    blurry_frames: int
    dark_frames: int
    overexposed_frames: int
    duplicate_frames: int
    median_sharpness: float
    viewpoint_change: float
    """Accumulated appearance decorrelation along the clip (0 = static view). Orbit, pan and
    forward flight all raise it; below ~0.15 there is too little parallax for 3D."""


class PreflightReport(BaseModel):
    schema_version: Literal[1] = PREFLIGHT_SCHEMA_VERSION
    asset_id: str
    source_sha256: str
    generated_at: datetime
    probe: VideoProbe
    analysis: AnalysisInfo
    summary: PreflightSummary
    keyframes: list[Keyframe]
    warnings: list[PreflightWarning]
    telemetry_present: bool
    settings: dict[str, float | int | str]
    series: FrameSeries
