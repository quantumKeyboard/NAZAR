"""Offline bundle mode (ADR-0003 fallback) for a GPU machine with no route to the API.

1. `export` (online, next to the API): lease one queued job and save its inputs (preflight,
   keyframes, telemetry) in a zip. No event is sent, so when the lease lapses (60 s by default)
   the job is simply queued again; no attempt is used up.
2. `run` (offline, on the GPU machine): the normal `JobRun` against `BundleClient`, which reads
   the inputs from the zip and records every event and artifact, in order, into a result zip.
3. `import` (online): lease that same job by id and replay the recording through the normal
   worker protocol. The API validates it exactly like a live worker's reports.

Neither zip contains the worker token or a live lease id. Stage durations are the real
offline ones; event timestamps are the import time.
"""

import hashlib
import json
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from nazar_contracts import (
    ArtifactCreate,
    ArtifactCreated,
    ComputeTarget,
    EventType,
    JobStatus,
    PipelineMode,
    WorkerEvent,
    WorkerEventAck,
    WorkerHeartbeat,
    WorkerLease,
)
from nazar_worker.client import WorkerClient
from nazar_worker.forecast import Forecast
from nazar_worker.profile import for_location
from nazar_worker.runner import JobRun, adapters_for, geometry_backend

BUNDLE_FORMAT = "nazar.mission-bundle/1"
RESULT_FORMAT = "nazar.result-bundle/1"
OFFLINE_LEASE_ID = "offline-bundle"
_ARTIFACT_NAME = re.compile(r"^artifacts/\d{3,}\.bin$")
_BATCH = 100
_HEARTBEAT_S = 10.0
_FINISHED = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class BundleError(Exception):
    """The bundle is unusable or does not match the job; nothing was sent for that job."""


def export_bundle(
    client: WorkerClient,
    worker_id: str,
    out: Path,
    modes: list[PipelineMode],
    job_id: str | None = None,
) -> WorkerLease | None:
    """Lease a queued job (the oldest, or `job_id`) and write its inputs to `out`.
    Returns None when there is no such job."""
    lease = client.lease(worker_id, modes, job_id=job_id)
    if lease is None:
        return None
    stored = lease.model_copy(update={"lease_id": OFFLINE_LEASE_ID})
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "bundle.json",
            json.dumps(
                {
                    "format": BUNDLE_FORMAT,
                    "exported_at": datetime.now(UTC).isoformat(),
                    "location": client.location.value,
                    "lease": json.loads(stored.model_dump_json()),
                },
                indent=2,
            ),
        )
        zf.writestr("preflight.json", json.dumps(client.preflight(lease.job_id)))
        for image in range(1, lease.keyframe_count + 1):
            # JPEGs do not compress further; store them.
            zf.writestr(
                f"keyframes/{image:04d}.jpg",
                client.keyframe(lease.job_id, image),
                compress_type=zipfile.ZIP_STORED,
            )
        if lease.telemetry_asset_id is not None:
            track = client.telemetry(lease.job_id)
            if track is not None:
                zf.writestr("telemetry.json", json.dumps(track))
    tmp.replace(out)
    return lease


class BundleClient:
    """`WorkerApi` backed by a mission bundle; records the run into a result zip."""

    def __init__(self, bundle: Path, out: Path) -> None:
        self._in = zipfile.ZipFile(bundle)
        try:
            meta = json.loads(self._in.read("bundle.json"))
        except (KeyError, ValueError) as exc:
            self._in.close()
            raise BundleError(f"{bundle.name} is not a NAZAR mission bundle.") from exc
        if meta.get("format") != BUNDLE_FORMAT:
            self._in.close()
            raise BundleError(f"Unsupported bundle format: {meta.get('format')!r}.")
        self.lease = WorkerLease.model_validate(meta["lease"])
        self.location = ComputeTarget(meta.get("location", ComputeTarget.LOCAL.value))
        self._out_path = out
        self._tmp = out.with_name(out.name + ".part")
        self._out = zipfile.ZipFile(self._tmp, "w", zipfile.ZIP_DEFLATED)
        self.ops: list[dict[str, Any]] = []
        self._artifacts = 0
        self._noted = False
        self._status = JobStatus.QUEUED

    def __enter__(self) -> "BundleClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._in.close()
        if not self._out.fp:  # already finished
            return
        self._out.close()
        self._tmp.unlink(missing_ok=True)

    def heartbeat(self, job_id: str) -> WorkerHeartbeat:
        return WorkerHeartbeat(cancel_requested=False, lease_expires_at=self.lease.lease_expires_at)

    def events(self, job_id: str, events: list[WorkerEvent]) -> WorkerEventAck:
        for event in events:
            self.ops.append({"op": "event", "event": json.loads(event.model_dump_json())})
            if event.stage is not None:
                self._status = JobStatus(event.stage.value)
            if event.type is EventType.STAGE_STARTED and not self._noted:
                # Said once, inside the first running stage, so the job page explains why
                # every event of this job arrived at once.
                self._noted = True
                note = WorkerEvent(
                    key=f"{event.stage.value if event.stage else 'job'}:warning:offline_bundle",
                    type=EventType.WARNING,
                    stage=event.stage,
                    data={
                        "code": "offline_bundle",
                        "message": "Processed offline from a mission bundle and imported "
                        "afterwards: stage durations are real, event times are the import time.",
                        "severity": "info",
                    },
                )
                self.ops.append({"op": "event", "event": json.loads(note.model_dump_json())})
        return WorkerEventAck(status=self._status, last_seq=len(self.ops), cancel_requested=False)

    def preflight(self, job_id: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self._in.read("preflight.json"))
        return data

    def keyframe(self, job_id: str, image: int) -> bytes:
        return self._in.read(f"keyframes/{image:04d}.jpg")

    def telemetry(self, job_id: str) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads(self._in.read("telemetry.json"))
        except KeyError:
            return None
        return data

    def publish(self, job_id: str, declared: ArtifactCreate, content: bytes) -> ArtifactCreated:
        self._artifacts += 1
        name = f"artifacts/{self._artifacts:03d}.bin"
        self._out.writestr(name, content)
        self.ops.append(
            {"op": "artifact", "file": name, "declared": json.loads(declared.model_dump_json())}
        )
        return ArtifactCreated(id=f"offline-{self._artifacts}", revision=1, uploaded=True)

    def finish(self, outcome: str, worker_id: str) -> None:
        self._out.writestr(
            "result.json",
            json.dumps(
                {
                    "format": RESULT_FORMAT,
                    "job_id": self.lease.job_id,
                    "pipeline": self.lease.pipeline.value,
                    "location": self.location.value,
                    "video_sha256": self.lease.video_sha256,
                    "worker_id": worker_id,
                    "geometry": (
                        g.value if (g := geometry_backend([self.lease.pipeline])) else None
                    ),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "outcome": outcome,
                    "ops": self.ops,
                },
                indent=1,
            ),
        )
        self._out.close()
        self._tmp.replace(self._out_path)


def run_bundle(bundle: Path, out: Path, worker_id: str, step_s: float = 0.0) -> str:
    """Process a mission bundle with no network. Returns the outcome (completed / failed)."""
    with BundleClient(bundle, out) as client:
        lease = client.lease
        adapters = adapters_for(lease.pipeline, step_s, client.location)
        forecast = (
            Forecast.for_worker(for_location(client.location))
            if lease.pipeline is PipelineMode.STANDARD
            else None
        )
        run = JobRun(client, lease, adapters, heartbeat_s=3600, forecast=forecast)
        outcome = run.execute()
        client.finish(outcome, worker_id)
    return outcome


def import_result(client: WorkerClient, result: Path, worker_id: str) -> str:
    """Replay a result bundle into the API as the job's worker. Returns the recorded outcome,
    or "cancelled" if the operator cancelled the job meanwhile."""
    with zipfile.ZipFile(result) as zf:
        try:
            meta = json.loads(zf.read("result.json"))
        except (KeyError, ValueError) as exc:
            raise BundleError(f"{result.name} is not a NAZAR result bundle.") from exc
        if meta.get("format") != RESULT_FORMAT:
            raise BundleError(f"Unsupported result format: {meta.get('format')!r}.")
        job_id = str(meta["job_id"])
        for op in meta["ops"]:  # check every artifact before anything is sent
            if op["op"] == "artifact":
                _check_artifact(zf, op)
        lease = client.lease(
            worker_id,
            [PipelineMode(meta["pipeline"])],
            job_id=job_id,
            location=ComputeTarget(meta.get("location", ComputeTarget.LOCAL.value)),
        )
        if lease is None:
            raise BundleError(
                f"Job {job_id} cannot be leased: it is still held by the export lease (wait a "
                "minute), was run by another worker, finished, or was cancelled."
            )
        if lease.video_sha256 != meta["video_sha256"]:
            raise BundleError("The result was made from a different video than this job's.")

        pending: list[WorkerEvent] = []
        last_beat = time.monotonic()

        def flush() -> bool:
            """Send buffered events; True if the operator cancelled the job."""
            nonlocal pending
            cancelled = False
            for i in range(0, len(pending), _BATCH):
                ack = client.events(job_id, pending[i : i + _BATCH])
                cancelled |= ack.cancel_requested and ack.status not in _FINISHED
            pending = []
            return cancelled

        for op in meta["ops"]:
            if time.monotonic() - last_beat > _HEARTBEAT_S:
                if client.heartbeat(job_id).cancel_requested:
                    return _cancel(client, job_id)
                last_beat = time.monotonic()
            if op["op"] == "event":
                pending.append(WorkerEvent.model_validate(op["event"]))
                continue
            if flush():
                return _cancel(client, job_id)
            declared, content = _check_artifact(zf, op)
            client.publish(job_id, declared, content)
        if flush():
            return _cancel(client, job_id)
    return str(meta["outcome"])


def _check_artifact(zf: zipfile.ZipFile, op: dict[str, Any]) -> tuple[ArtifactCreate, bytes]:
    declared = ArtifactCreate.model_validate(op["declared"])
    name = str(op["file"])
    try:
        size = zf.getinfo(name).file_size
    except KeyError:
        size = -1
    if not _ARTIFACT_NAME.match(name) or size != declared.byte_size:
        raise BundleError(f"Artifact {name} does not match its declaration.")
    content = zf.read(name)
    if hashlib.sha256(content).hexdigest() != declared.checksum_sha256:
        raise BundleError(f"Artifact {name} is corrupted (checksum mismatch).")
    return declared, content


def _cancel(client: WorkerClient, job_id: str) -> str:
    client.events(job_id, [WorkerEvent(key="job:cancelled", type=EventType.JOB_CANCELLED)])
    return "cancelled"
