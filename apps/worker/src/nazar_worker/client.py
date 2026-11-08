"""Outbound-only HTTP client for the worker protocol (ADR-0003). Standard library only.

Every call is safe to repeat (events and artifacts carry idempotency keys), so transient
network errors and 5xx responses are retried with backoff. A 409 means the lease was lost.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel

from nazar_contracts import (
    ArtifactCreate,
    ArtifactCreated,
    ComputeTarget,
    GeometryBackend,
    ModelReadiness,
    PipelineMode,
    TimeModel,
    WorkerEvent,
    WorkerEventAck,
    WorkerEventBatch,
    WorkerHeartbeat,
    WorkerLease,
    WorkerLeaseRequest,
)

Transport = Callable[[str, str, bytes | None, dict[str, str]], tuple[int, bytes]]
"""(method, url, body, headers) -> (status, body). Swappable so tests can run in-process."""


class WorkerApi(Protocol):
    """What the job runner needs from the API. `WorkerClient` speaks HTTP; the offline
    `BundleClient` (bundle.py) reads inputs from a mission bundle and records the outputs."""

    def heartbeat(self, job_id: str) -> WorkerHeartbeat: ...
    def events(self, job_id: str, events: list[WorkerEvent]) -> WorkerEventAck: ...
    def preflight(self, job_id: str) -> dict[str, Any]: ...
    def keyframe(self, job_id: str, image: int) -> bytes: ...
    def telemetry(self, job_id: str) -> dict[str, Any] | None: ...
    def publish(self, job_id: str, declared: ArtifactCreate, content: bytes) -> ArtifactCreated: ...


class LeaseLost(Exception):
    """The API no longer recognises this worker's lease; drop the job without reporting."""


class ApiCallFailed(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class WorkerClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        attempts: int = 4,
        timeout_s: float = 60.0,
        transport: Transport | None = None,
        location: ComputeTarget = ComputeTarget.LOCAL,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("The API URL must start with http:// or https://")
        self._base = base_url.rstrip("/")
        self._token = token
        self._attempts = attempts
        self._timeout = timeout_s
        self._transport = transport or self._urllib
        self.lease_id: str | None = None
        self.location = location
        """Where this worker runs; it only leases jobs started for this location."""

    def _urllib(
        self, method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        # The scheme is restricted to http(s) in __init__.
        req = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> tuple[int, bytes]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Nazar-Request": "1",
            "X-Nazar-Worker-Location": self.location.value,
            "Content-Type": content_type,
        }
        if self.lease_id:
            headers["X-Nazar-Lease"] = self.lease_id
        delay = 0.5
        for attempt in range(1, self._attempts + 1):
            try:
                status, data = self._transport(method, self._base + path, body, headers)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                if attempt == self._attempts:
                    raise ApiCallFailed(0, f"API unreachable: {err}") from err
            else:
                if status < 400:
                    return status, data
                detail = _detail(data)
                if status == 409 and "Lease lost" in detail:
                    raise LeaseLost(detail)
                if status < 500 or attempt == self._attempts:
                    raise ApiCallFailed(status, detail)
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def _json(self, method: str, path: str, model: BaseModel | None = None) -> Any:
        body = model.model_dump_json().encode() if model is not None else None
        status, data = self._request(method, path, body)
        return None if status == 204 or not data else json.loads(data)

    def lease(
        self,
        worker_id: str,
        modes: list[PipelineMode],
        geometry: GeometryBackend | None = None,
        job_id: str | None = None,
        location: ComputeTarget | None = None,
        models: list[ModelReadiness] | None = None,
        time_model: TimeModel | None = None,
    ) -> WorkerLease | None:
        """`location` overrides this client's own (bundle import acts for the college)."""
        self.lease_id = None
        body = WorkerLeaseRequest(
            worker_id=worker_id,
            modes=modes,
            geometry=geometry,
            job_id=job_id,
            location=location or self.location,
            models=models,
            time_model=time_model,
        )
        got = self._json("POST", "/api/worker/lease", body)
        if got is None:
            return None
        lease = WorkerLease.model_validate(got)
        self.lease_id = lease.lease_id
        return lease

    def heartbeat(self, job_id: str) -> WorkerHeartbeat:
        return WorkerHeartbeat.model_validate(
            self._json("POST", f"/api/worker/jobs/{job_id}/heartbeat")
        )

    def events(self, job_id: str, events: list[WorkerEvent]) -> WorkerEventAck:
        batch = WorkerEventBatch(events=events)
        return WorkerEventAck.model_validate(
            self._json("POST", f"/api/worker/jobs/{job_id}/events", batch)
        )

    def preflight(self, job_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._json("GET", f"/api/worker/jobs/{job_id}/inputs/preflight")
        return result

    def keyframe(self, job_id: str, image: int) -> bytes:
        """One extracted keyframe JPEG (`image` is the 1-based number from the preflight)."""
        _, data = self._request("GET", f"/api/worker/jobs/{job_id}/inputs/keyframes/{image}.jpg")
        return data

    def telemetry(self, job_id: str) -> dict[str, Any] | None:
        """The job's normalised telemetry track, or None if it was removed after job creation."""
        try:
            result: dict[str, Any] = self._json(
                "GET", f"/api/worker/jobs/{job_id}/inputs/telemetry"
            )
        except ApiCallFailed as exc:
            if exc.status == 404:
                return None
            raise
        return result

    def publish(self, job_id: str, declared: ArtifactCreate, content: bytes) -> ArtifactCreated:
        created = ArtifactCreated.model_validate(
            self._json("POST", f"/api/worker/jobs/{job_id}/artifacts", declared)
        )
        if created.uploaded:
            return created
        _, data = self._request(
            "PUT",
            f"/api/worker/jobs/{job_id}/artifacts/{created.id}/content",
            content,
            "application/octet-stream",
        )
        return ArtifactCreated.model_validate_json(data)


def _detail(raw: bytes) -> str:
    try:
        detail = json.loads(raw).get("detail")
    except (ValueError, AttributeError):
        return raw[:200].decode(errors="replace")
    return detail if isinstance(detail, str) else json.dumps(detail)[:300]
