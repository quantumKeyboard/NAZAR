"""The worker job loop: lease a job, run its stages in order, report events, keep the lease.

Event keys are deterministic per (stage, step), so after a crash or a dropped response the
worker can resend and the API ignores duplicates. If the lease is lost (the API re-queued
the job), the worker drops the job silently: another attempt now owns it.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from nazar_contracts import (
    PIPELINE_STAGES,
    ArtifactCreate,
    ArtifactCreated,
    ComputeTarget,
    EventType,
    GeometryBackend,
    ModelReadiness,
    PipelineMode,
    StageName,
    TimeModel,
    WorkerEvent,
    WorkerLease,
)
from nazar_worker.client import ApiCallFailed, LeaseLost, WorkerApi, WorkerClient
from nazar_worker.fast3r import Fast3rRuntime
from nazar_worker.forecast import Forecast, model_readiness
from nazar_worker.gpu import GpuRuntime
from nazar_worker.profile import for_location
from nazar_worker.stages import (
    Cancelled,
    StageAdapter,
    StageContext,
    StageFailed,
    WarningSeverity,
)
from nazar_worker.standard import build_standard_pipeline
from nazar_worker.test_pipeline import build_test_pipeline

log = logging.getLogger("nazar_worker")


def adapters_for(
    mode: PipelineMode, step_s: float, location: ComputeTarget = ComputeTarget.LOCAL
) -> dict[StageName, StageAdapter]:
    """The worker's location picks the compute profile: the college worker runs `clg`."""
    if mode is PipelineMode.TEST:
        return build_test_pipeline(step_s)
    return build_standard_pipeline(profile=for_location(location))


class _Heartbeat(threading.Thread):
    """Renews the lease in the background and records cancel requests / lease loss."""

    def __init__(self, client: WorkerApi, job_id: str, interval_s: float) -> None:
        super().__init__(daemon=True, name="nazar-heartbeat")
        self.client, self.job_id, self.interval_s = client, job_id, interval_s
        self.cancel = threading.Event()
        self.lost = threading.Event()
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                if self.client.heartbeat(self.job_id).cancel_requested:
                    self.cancel.set()
            except LeaseLost:
                self.lost.set()
                return
            except ApiCallFailed as exc:
                log.warning("heartbeat failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()


class JobRun:
    def __init__(
        self,
        client: WorkerApi,
        lease: WorkerLease,
        adapters: dict[StageName, StageAdapter],
        heartbeat_s: float,
        forecast: Forecast | None = None,
    ) -> None:
        self.client, self.lease, self.adapters = client, lease, adapters
        self.heartbeat = _Heartbeat(client, lease.job_id, heartbeat_s)
        self.forecast = forecast
        self._progress_n = 0
        self._gpu: dict[StageName, tuple[int, int]] = {}

    def send(
        self,
        key: str,
        type_: EventType,
        stage: StageName | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        ack = self.client.events(
            self.lease.job_id, [WorkerEvent(key=key, type=type_, stage=stage, data=data)]
        )
        if ack.cancel_requested:
            self.heartbeat.cancel.set()

    def check_cancel(self) -> None:
        if self.heartbeat.lost.is_set():
            raise LeaseLost("lease lost during stage")
        if self.heartbeat.cancel.is_set():
            raise Cancelled

    def _context(self, stage: StageName) -> StageContext:
        def progress(message: str, done: int | None, total: int | None, unit: str | None) -> None:
            self._progress_n += 1
            self.send(
                f"{stage.value}:progress:{self._progress_n}",
                EventType.STAGE_PROGRESS,
                stage,
                {"message": message, "completed": done, "total": total, "unit": unit},
            )

        def warn(code: str, message: str, severity: WarningSeverity) -> None:
            self.send(
                f"{stage.value}:warning:{code}",
                EventType.WARNING,
                stage,
                {"code": code, "message": message, "severity": severity},
            )

        def publish(declared: ArtifactCreate, content: bytes) -> ArtifactCreated:
            self.check_cancel()
            return self.client.publish(self.lease.job_id, declared, content)

        def gpu(peak: int | None, budget: int | None) -> None:
            if peak is not None:
                old = self._gpu.get(stage, (0, 0))[0]
                self._gpu[stage] = (max(old, peak), budget or 0)

        def preflight() -> dict[str, Any]:
            return self.client.preflight(self.lease.job_id)

        def keyframe(image: int) -> bytes:
            return self.client.keyframe(self.lease.job_id, image)

        def telemetry() -> dict[str, Any] | None:
            if self.lease.telemetry_asset_id is None:
                return None
            return self.client.telemetry(self.lease.job_id)

        return StageContext(
            lease=self.lease,
            progress=progress,
            warn=warn,
            publish=publish,
            preflight=preflight,
            keyframe=keyframe,
            check_cancel=self.check_cancel,
            telemetry=telemetry,
            gpu=gpu,
        )

    def execute(self) -> str:
        """Run every stage. Returns the outcome: completed, failed, cancelled or lost."""
        self.heartbeat.start()
        try:
            self._send_estimate()
            for stage in PIPELINE_STAGES:
                self.check_cancel()
                adapter = self.adapters.get(stage)
                if adapter is None:
                    if stage in (StageName.GEOMETRY, StageName.DENSE_RECONSTRUCTION):
                        self.send(
                            "job:failed",
                            EventType.JOB_FAILED,
                            data={
                                "reason": f"The {stage.value} stage has no implementation in "
                                "this worker yet, so no reconstruction can be produced.",
                                "retryable": False,
                            },
                        )
                        return "failed"
                    self._skip(stage, "Not implemented in this worker yet.")
                    continue
                self._run_stage(stage, adapter)
            self.send("job:completed", EventType.JOB_COMPLETED, data={"artifact_ids": []})
            return "completed"
        except Cancelled:
            self.send("job:cancelled", EventType.JOB_CANCELLED)
            return "cancelled"
        except LeaseLost:
            log.warning("lease lost; dropping job %s", self.lease.job_id)
            return "lost"
        except StageFailed as exc:
            log.info("job %s cannot be reconstructed: %s", self.lease.job_id, exc)
            return self._fail(str(exc), exc.retryable)
        except Exception as exc:
            log.exception("job %s failed", self.lease.job_id)
            try:
                self.send(
                    "job:failed",
                    EventType.JOB_FAILED,
                    data={"reason": _reason(exc), "retryable": True},
                )
            except (LeaseLost, ApiCallFailed):
                return "lost"
            return "failed"
        finally:
            self.heartbeat.stop()

    def _send_estimate(self) -> None:
        """Tell the UI how long each stage should take. A forecast problem never fails a job."""
        if self.forecast is None:
            return
        try:
            estimate = self.forecast.estimate(
                self.lease.keyframe_count, self.lease.telemetry_asset_id is not None
            )
            self.send("job:estimate", EventType.JOB_ESTIMATE, data=estimate.model_dump(mode="json"))
        except (ApiCallFailed, ValueError) as exc:
            log.warning("could not send the time estimate: %s", exc)

    def _fail(self, reason: str, retryable: bool) -> str:
        try:
            self.send(
                "job:failed",
                EventType.JOB_FAILED,
                data={"reason": reason[:500], "retryable": retryable},
            )
        except (LeaseLost, ApiCallFailed):
            return "lost"
        return "failed"

    def _skip(self, stage: StageName, reason: str) -> None:
        self.send(f"{stage.value}:skipped", EventType.STAGE_SKIPPED, stage, {"reason": reason})

    def _run_stage(self, stage: StageName, adapter: StageAdapter) -> None:
        reason = adapter.skip_reason(self.lease)
        if reason is not None:
            self._skip(stage, reason)
            return
        started = time.monotonic()
        try:
            self.send(f"{stage.value}:started", EventType.STAGE_STARTED, stage)
        except ApiCallFailed as exc:
            # The user may cancel between our last check and this start; the API then refuses
            # the new stage. That is a cancellation, not a worker error.
            if exc.status == 409 and self.client.heartbeat(self.lease.job_id).cancel_requested:
                raise Cancelled from exc
            raise
        adapter.run(self._context(stage))
        seconds = time.monotonic() - started
        data: dict[str, Any] = {"duration_s": round(seconds, 3)}
        if stage in self._gpu:
            data["peak_vram_mib"], data["budget_mib"] = self._gpu[stage]
        self.send(f"{stage.value}:completed", EventType.STAGE_COMPLETED, stage, data)
        if self.forecast is not None:
            self.forecast.record(stage, self.lease.keyframe_count, seconds)


def _reason(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return f"Worker error: {text}"[:500]


def _models_report(modes: list[PipelineMode], client: WorkerClient) -> list[ModelReadiness] | None:
    """The models this worker can run, sent with every lease request so the UI can show them."""
    if PipelineMode.STANDARD not in modes:
        return None
    return model_readiness(for_location(client.location), GpuRuntime.from_env())


def _time_model(modes: list[PipelineMode], client: WorkerClient) -> TimeModel | None:
    if PipelineMode.STANDARD not in modes:
        return None
    return Forecast.for_worker(for_location(client.location)).time_model()


def geometry_backend(modes: list[PipelineMode]) -> GeometryBackend | None:
    """Reported on every lease so readiness shows what standard jobs would use."""
    if PipelineMode.STANDARD not in modes:
        return None
    usable = isinstance(Fast3rRuntime.from_env(), Fast3rRuntime)
    return GeometryBackend.FAST3R if usable else GeometryBackend.CPU_BASELINE


def run_loop(
    client: WorkerClient,
    worker_id: str,
    modes: list[PipelineMode],
    *,
    step_s: float = 0.0,
    poll_s: float = 2.0,
    heartbeat_s: float = 10.0,
    once: bool = False,
    stop: Callable[[], bool] = lambda: False,
) -> int:
    """Lease and run jobs until `stop()` is true. With `once`, return after one job (or
    immediately if the queue is empty). Returns the number of jobs handled."""
    handled = 0
    log.info("worker %s polling for %s jobs", worker_id, ",".join(m.value for m in modes))
    while not stop():
        try:
            lease = client.lease(
                worker_id,
                modes,
                geometry_backend(modes),
                models=_models_report(modes, client),
                time_model=_time_model(modes, client),
            )
        except ApiCallFailed as exc:
            log.warning("lease request failed: %s", exc)
            lease = None
        if lease is None:
            if once:
                return handled
            time.sleep(poll_s)
            continue
        log.info("leased job %s (attempt %s, %s)", lease.job_id, lease.attempt, lease.pipeline)
        adapters = adapters_for(lease.pipeline, step_s, client.location)
        forecast = (
            Forecast.for_worker(for_location(client.location))
            if lease.pipeline is PipelineMode.STANDARD
            else None
        )
        outcome = JobRun(client, lease, adapters, heartbeat_s, forecast).execute()
        log.info("job %s %s", lease.job_id, outcome)
        handled += 1
        if once:
            return handled
    return handled
