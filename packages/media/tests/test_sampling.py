"""Frame sampling at a chosen rate."""

import pytest

from nazar_contracts import AUTO_MIN_FRAMES, FrameFlag, FrameSeries, KeyframeReason
from nazar_media import MediaError
from nazar_media.sampling import auto_frames, count_frames, plan_frames


def series(
    n: int,
    fps: float = 10.0,
    sharp: list[float] | None = None,
    flags: dict[int, list[FrameFlag]] | None = None,
) -> FrameSeries:
    return FrameSeries(
        index=list(range(n)),
        t=[i / fps for i in range(n)],
        sharpness=sharp or [10.0] * n,
        brightness=[0.5] * n,
        change=[0.01] * n,
        motion_px=[1.0] * n,
        flags=[(flags or {}).get(i, []) for i in range(n)],
    )


def test_one_frame_per_second() -> None:
    plan = plan_frames(series(101), fps=1.0, cap=64)  # a 10 s video
    assert len(plan.keyframes) == 10
    assert not plan.clamped
    assert all(k.reason is KeyframeReason.SAMPLED for k in plan.keyframes)
    assert [k.image for k in plan.keyframes] == list(range(1, 11))
    assert plan.fps_effective == pytest.approx(1.0, abs=0.01)


def test_each_window_takes_its_sharpest_central_frame() -> None:
    sharp = [1.0] * 101
    sharp[5] = 50.0  # centre of window 0 (0-1 s)
    sharp[1] = 99.0  # sharper, but at the edge of the window: not preferred over the centre
    plan = plan_frames(series(101, sharp=sharp), fps=1.0, cap=64)
    assert plan.keyframes[0].index == 5


def test_unusable_frames_are_never_picked() -> None:
    sharp = [1.0] * 101
    sharp[5] = 100.0
    plan = plan_frames(series(101, sharp=sharp, flags={5: [FrameFlag.BLURRY]}), fps=1.0, cap=64)
    assert plan.keyframes[0].index != 5


def test_a_window_with_no_usable_frame_is_dropped() -> None:
    flags = {i: [FrameFlag.DARK] for i in range(20, 30)}  # all of window 2
    plan = plan_frames(series(101, flags=flags), fps=1.0, cap=64)
    assert plan.dropped == 1
    assert len(plan.keyframes) == 9


def test_the_cap_lowers_the_rate_and_says_so() -> None:
    plan = plan_frames(series(1001), fps=4.0, cap=64)  # a 100 s video wants 400 frames
    assert plan.clamped
    assert plan.wanted == 400
    assert len(plan.keyframes) == 64
    assert plan.fps_effective == pytest.approx(0.64, abs=0.01)


def test_too_few_frames_is_an_error_with_advice() -> None:
    with pytest.raises(MediaError, match="Choose a higher rate"):
        plan_frames(series(11), fps=0.25, cap=64)  # a 1 s video at 0.25 fps


def test_count_frames_is_bounded_by_decoded_frames() -> None:
    assert count_frames(38.4, 1.0, 64, 1000) == (38, 38, False)
    assert count_frames(2.0, 4.0, 64, 5) == (8, 5, False)
    assert count_frames(100.0, 1.0, 64, 3000) == (100, 64, True)


def test_automatic_mode_samples_a_short_clip_densely() -> None:
    # A 6.6 s clip: 1 fps would give 7 views; automatic asks for the minimum instead.
    assert auto_frames(6.6, 40) == AUTO_MIN_FRAMES
    plan = plan_frames(series(661, fps=100.0), fps=None, cap=64, auto_max=40)
    assert plan.auto
    assert len(plan.keyframes) == AUTO_MIN_FRAMES
    assert plan.fps_effective > 3


def test_automatic_mode_is_about_one_per_second_for_a_long_clip_up_to_its_maximum() -> None:
    assert auto_frames(30.0, 40) == 30
    assert auto_frames(300.0, 40) == 40
    assert auto_frames(300.0, 96) == 96


def test_automatic_mode_is_still_bounded_by_the_decoded_frames() -> None:
    assert count_frames(6.0, None, 64, 10, auto_max=40) == (AUTO_MIN_FRAMES, 10, False)
