"""The time forecast and model readiness."""

from pathlib import Path

import pytest

from nazar_contracts import StageName
from nazar_worker.forecast import Forecast, model_readiness
from nazar_worker.gpu import GpuRuntime
from nazar_worker.profile import CLG, LOCAL

ALL_READY = {s: True for s in StageName}


def forecast(profile=LOCAL, ready=None, cpu=False, store=None) -> Forecast:  # type: ignore[no-untyped-def]
    return Forecast(profile, ready or dict(ALL_READY), cpu_baseline=cpu, store=store)


def test_default_estimate_matches_the_measured_local_run() -> None:
    est = forecast().estimate(22, telemetry=False)
    # sample1, 22 frames, idle RTX 4050: ~6 s masks, ~26 s Fast3R, ~54 s depth, ~5 min splat.
    assert est.stages[StageName.DYNAMIC_ANALYSIS] == pytest.approx(6.3, abs=1.5)
    assert est.stages[StageName.GEOMETRY] == pytest.approx(26, abs=4)
    assert est.stages[StageName.DEPTH_PRIORS] == pytest.approx(53, abs=6)
    assert 270 < est.stages[StageName.DENSE_RECONSTRUCTION] < 330
    # The height-map surface + texture bake is NOT yet measured (the old Poisson step was
    # ~22 s); this pins the current estimate until a real run calibrates it.
    assert 40 < est.stages[StageName.SURFACE_PROCESSING] < 130
    assert est.stages[StageName.SENSOR_OPTIMIZATION] == 0  # no telemetry: skipped
    assert 6 * 60 < est.total_s < 9.5 * 60
    assert "RTX 4050" in est.basis


def test_more_frames_cost_more_but_the_splat_does_not_grow() -> None:
    f = forecast()
    few, many = f.estimate(20, False), f.estimate(60, False)
    assert many.stages[StageName.DEPTH_PRIORS] > 2.5 * few.stages[StageName.DEPTH_PRIORS]
    splat_few = few.stages[StageName.DENSE_RECONSTRUCTION]
    splat_many = many.stages[StageName.DENSE_RECONSTRUCTION]
    assert splat_many < splat_few * 1.1  # iterations, not frames, drive the splat


def test_telemetry_adds_the_sensor_stage() -> None:
    assert forecast().estimate(22, True).stages[StageName.SENSOR_OPTIMIZATION] > 0


def test_a_model_that_is_not_installed_costs_nothing_on_local() -> None:
    ready = {**ALL_READY, StageName.DEPTH_PRIORS: False}
    est = forecast(ready=ready).estimate(22, False)
    assert est.stages[StageName.DEPTH_PRIORS] == 0


def test_the_cpu_baseline_skips_the_gpu_only_stages() -> None:
    est = forecast(cpu=True).estimate(22, False)
    for stage in (StageName.DYNAMIC_ANALYSIS, StageName.DEPTH_PRIORS, StageName.SURFACE_PROCESSING):
        assert est.stages[stage] == 0
    assert "CPU baseline" in est.basis


def test_college_estimate_says_it_is_unmeasured() -> None:
    est = forecast(CLG).estimate(22, False)
    assert "not yet measured" in est.basis


def test_measured_runs_calibrate_later_estimates(tmp_path: Path) -> None:
    store = tmp_path / "timings.json"
    f = forecast(store=store)
    base = f.estimate(22, False)
    # This machine is twice as slow as the default model on every stage that ran.
    for stage in StageName:
        planned = base.stages[stage]
        if planned > 0:
            f.record(stage, 22, planned * 2)
    again = f.estimate(22, False)
    assert again.stages[StageName.GEOMETRY] == pytest.approx(
        2 * base.stages[StageName.GEOMETRY], rel=0.01
    )
    assert again.total_s == pytest.approx(2 * base.total_s, rel=0.02)
    assert "Calibrated from earlier jobs" in again.basis


def test_one_slow_run_among_good_ones_does_not_move_the_estimate(tmp_path: Path) -> None:
    f = forecast(store=tmp_path / "t.json")
    planned = f.estimate(22, False).stages[StageName.GEOMETRY]
    for seconds in (planned, planned, planned * 3, planned, planned):
        f.record(StageName.GEOMETRY, 22, seconds)
    assert f.estimate(22, False).stages[StageName.GEOMETRY] == pytest.approx(planned, rel=0.01)


def test_a_corrupt_history_file_is_ignored(tmp_path: Path) -> None:
    store = tmp_path / "t.json"
    store.write_text("{not json", encoding="utf-8")
    assert forecast(store=store).estimate(22, False).total_s > 0


def test_readiness_names_what_is_missing(tmp_path: Path) -> None:
    gpu = GpuRuntime(main_python=None, splat_python=None, weights=tmp_path, root=tmp_path)
    rows = {m.name: m for m in model_readiness(LOCAL, gpu)}
    assert not rows["YOLO"].ready
    assert "not set up" in rows["YOLO"].detail
    assert next(m.stage for m in rows.values()) is StageName.DYNAMIC_ANALYSIS


def test_runs_without_a_splat_do_not_calibrate_a_run_with_one(tmp_path: Path) -> None:
    """Regression: 0.06 s 'dense' stages from splat-less runs halved the real estimate."""
    store = tmp_path / "t.json"
    no_splat = forecast(ready={**ALL_READY, StageName.DENSE_RECONSTRUCTION: False}, store=store)
    for _ in range(3):
        no_splat.record(StageName.DENSE_RECONSTRUCTION, 22, 0.06)
    full = forecast(store=store)
    planned = full.estimate(22, False).stages[StageName.DENSE_RECONSTRUCTION]
    for seconds in (300.0, 330.0):
        full.record(StageName.DENSE_RECONSTRUCTION, 22, seconds)
    after = full.estimate(22, False).stages[StageName.DENSE_RECONSTRUCTION]
    assert 290 < after < 340, (planned, after)


def test_implausible_old_timings_are_ignored(tmp_path: Path) -> None:
    store = tmp_path / "t.json"
    f = forecast(store=store)
    base = f.estimate(22, False).stages[StageName.DENSE_RECONSTRUCTION]
    for _ in range(4):
        f.record(StageName.DENSE_RECONSTRUCTION, 22, 0.06)
    assert f.estimate(22, False).stages[StageName.DENSE_RECONSTRUCTION] == base
