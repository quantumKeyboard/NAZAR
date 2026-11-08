"""Frame sampling at a chosen rate (frames per second of video), or automatically.

The upload-time preflight picks keyframes by quality and viewpoint change. A reconstruction
job instead samples the video at the rate the user chose: the video is cut into equal time
windows (`duration * fps` of them, at most `cap`), and each window contributes its sharpest
usable frame, preferring the middle of the window so neighbours stay evenly spaced. A window
with no usable frame (all blurry, dark or over-exposed) is dropped, not filled with a bad one.
The result reuses the preflight report with its keyframes replaced, so everything downstream
(the worker, offline bundles) reads it the same way.
"""

import json
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from nazar_contracts import (
    AUTO_MIN_FRAMES,
    MIN_FRAMES,
    FrameFlag,
    FrameSeries,
    Keyframe,
    KeyframeReason,
    PreflightReport,
)
from nazar_media.ffmpeg import MediaError
from nazar_media.preflight import extract_keyframe_images

_UNUSABLE = frozenset({FrameFlag.BLURRY, FrameFlag.DARK, FrameFlag.OVEREXPOSED})
FRAMES_REPORT = "preflight.json"


@dataclass(frozen=True)
class SamplePlan:
    keyframes: list[Keyframe]
    wanted: int
    """Frames the requested rate asks for."""
    windows: int
    """Time windows used: `wanted`, lowered to the cap and to the decoded frame count."""
    clamped: bool
    dropped: int
    """Windows with no usable frame."""
    fps_requested: float
    fps_effective: float
    cap: int
    auto: bool = False
    """The frame count was chosen from the video length, not from a rate."""


def auto_frames(duration_s: float, auto_max: int) -> int:
    """Frames automatic mode asks for: about one per second of video, but never fewer than
    `AUTO_MIN_FRAMES` (a 6 s clip is sampled at ~4 fps, not reduced to 6 views) and never
    more than `auto_max`. Overlap between neighbouring frames is what the reconstruction
    needs, and a short clip has little of it to spare."""
    return min(auto_max, max(AUTO_MIN_FRAMES, round(duration_s)))


def count_frames(
    duration_s: float,
    fps: float | None,
    cap: int,
    decoded: int,
    auto_max: int | None = None,
) -> tuple[int, int, bool]:
    """(frames wanted, windows used, clamped) for a video of `duration_s` seconds. The caller
    uses this before any frame is read, so the UI can quote the number up front. `fps=None`
    is automatic mode (see `auto_frames`)."""
    wanted = (
        auto_frames(duration_s, auto_max if auto_max is not None else cap)
        if fps is None
        else max(1, round(duration_s * fps))
    )
    windows = min(wanted, cap, max(decoded, 1))
    return wanted, windows, wanted > cap


def plan_frames(
    series: FrameSeries, fps: float | None, cap: int, auto_max: int | None = None
) -> SamplePlan:
    t = series.t
    if len(t) < 2 or t[-1] - t[0] <= 0:
        raise MediaError("too_short", "The video is too short to sample frames from.")
    t0, duration = t[0], t[-1] - t[0]
    wanted, windows, clamped = count_frames(duration, fps, cap, len(t), auto_max)
    if windows < MIN_FRAMES:
        raise MediaError(
            "too_few_frames",
            f"At {fps:g} frames per second this {duration:.1f} s video gives {windows} "
            f"frame(s); at least {MIN_FRAMES} are needed. Choose a higher rate."
            if fps is not None
            else f"This {duration:.1f} s video has only {windows} distinct frame(s); at least "
            f"{MIN_FRAMES} are needed.",
        )
    median = statistics.median(series.sharpness) or 1.0
    usable = [not (_UNUSABLE & set(f)) for f in series.flags]
    width = duration / windows
    picks: list[Keyframe] = []
    for k in range(windows):
        lo, hi = t0 + k * width, t0 + (k + 1) * width
        a = bisect_left(t, lo)
        b = bisect_right(t, hi) if k == windows - 1 else bisect_left(t, hi)
        centre = (lo + hi) / 2
        core = [i for i in range(a, b) if abs(t[i] - centre) <= width / 4 and usable[i]]
        pool = core or [i for i in range(a, b) if usable[i]]
        if not pool:
            continue
        best = max(pool, key=lambda i: series.sharpness[i])
        picks.append(
            Keyframe(
                index=series.index[best],
                t=round(t[best], 4),
                reason=KeyframeReason.SAMPLED,
                quality=round(min(1.0, series.sharpness[best] / (1.5 * median)), 4),
                image=len(picks) + 1,
            )
        )
    if len(picks) < MIN_FRAMES:
        raise MediaError(
            "too_few_usable_frames",
            f"Only {len(picks)} usable frames were found "
            + (f"at {fps:g} frames per second " if fps is not None else "")
            + "(the rest are blurry, dark or over-exposed).",
        )
    return SamplePlan(
        keyframes=picks,
        wanted=wanted,
        windows=windows,
        clamped=clamped,
        dropped=windows - len(picks),
        fps_requested=fps if fps is not None else round(wanted / duration, 3),
        fps_effective=round(len(picks) / duration, 3),
        cap=cap,
        auto=fps is None,
    )


def sample_job_frames(
    src: Path,
    report: PreflightReport,
    fps: float | None,
    cap: int,
    out_dir: Path,
    timeout: float = 900,
    auto_max: int | None = None,
) -> SamplePlan:
    """Choose the frames, extract them at full resolution into `out_dir/keyframes`, and write
    the job's own `preflight.json` (the upload's report with these keyframes)."""
    plan = plan_frames(report.series, fps, cap, auto_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_keyframe_images(src, plan.keyframes, out_dir, timeout)
    sampled = report.model_copy(update={"keyframes": plan.keyframes})
    tmp = out_dir / (FRAMES_REPORT + ".tmp")
    tmp.write_text(json.dumps(json.loads(sampled.model_dump_json())), encoding="utf-8")
    tmp.replace(out_dir / FRAMES_REPORT)
    return plan
