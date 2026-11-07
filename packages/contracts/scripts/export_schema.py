"""Export the Pydantic contracts to JSON so TypeScript types can be generated from them.

Usage: uv run python packages/contracts/scripts/export_schema.py [--check]
--check exits non-zero if the committed schema differs from the models (CI drift guard).
"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nazar_contracts import (
    AUTO_MAX_FRAMES,
    AUTO_MIN_FRAMES,
    EVENT_SCHEMA_VERSION,
    FPS_DEFAULT,
    FPS_MAX,
    FPS_MIN,
    MAX_FRAMES,
    MIN_FRAMES,
    PIPELINE_STAGES,
    TERMINAL_STATUSES,
    ArtifactCreate,
    ArtifactCreated,
    ArtifactManifest,
    Asset,
    AssetCreate,
    AuthState,
    Credentials,
    EventType,
    HealthResponse,
    Job,
    JobEvent,
    JobStatus,
    Mission,
    MissionCreate,
    PreflightReport,
    ReadinessResponse,
    StageName,
    TelemetrySummary,
    WorkerEventAck,
    WorkerEventBatch,
    WorkerHeartbeat,
    WorkerLease,
    WorkerLeaseRequest,
)
from nazar_contracts.jobs import ALLOWED_TRANSITIONS

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"


class NazarContracts(BaseModel):
    """Bundle that pulls every public contract into one schema document."""

    health: HealthResponse
    readiness: ReadinessResponse
    job_event: JobEvent
    artifact_manifest: ArtifactManifest
    job_status: JobStatus
    stage_name: StageName
    event_type: EventType
    auth_state: AuthState
    credentials: Credentials
    mission: Mission
    mission_create: MissionCreate
    asset: Asset
    asset_create: AssetCreate
    preflight_report: PreflightReport
    telemetry_summary: TelemetrySummary
    job: Job
    artifact_create: ArtifactCreate
    artifact_created: ArtifactCreated
    worker_lease_request: WorkerLeaseRequest
    worker_lease: WorkerLease
    worker_heartbeat: WorkerHeartbeat
    worker_event_batch: WorkerEventBatch
    worker_event_ack: WorkerEventAck


def _strip_property_titles(schema: dict[str, Any]) -> None:
    """Drop per-field titles inside models so generated TS names models, not every field."""
    for node in schema.get("$defs", {}).values():
        for prop in node.get("properties", {}).values():
            prop.pop("title", None)


def render() -> dict[str, str]:
    schema = NazarContracts.model_json_schema(mode="serialization")
    _strip_property_titles(schema)
    constants = {
        "eventSchemaVersion": EVENT_SCHEMA_VERSION,
        "pipelineStages": [s.value for s in PIPELINE_STAGES],
        "fps": {"min": FPS_MIN, "max": FPS_MAX, "default": FPS_DEFAULT},
        "frames": {
            "min": MIN_FRAMES,
            "max": {k.value: v for k, v in MAX_FRAMES.items()},
            "auto_min": AUTO_MIN_FRAMES,
            "auto_max": {k.value: v for k, v in AUTO_MAX_FRAMES.items()},
        },
        "terminalStatuses": sorted(s.value for s in TERMINAL_STATUSES),
        "allowedTransitions": {
            k.value: sorted(t.value for t in v) for k, v in ALLOWED_TRANSITIONS.items()
        },
    }
    return {
        "contracts.schema.json": json.dumps(schema, indent=2, sort_keys=True) + "\n",
        "constants.json": json.dumps(constants, indent=2, sort_keys=True) + "\n",
    }


def main() -> int:
    check = "--check" in sys.argv
    SCHEMA_DIR.mkdir(exist_ok=True)
    stale: list[str] = []
    for name, content in render().items():
        path = SCHEMA_DIR / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        print(f"Contract schema out of date: {', '.join(stale)}. Run `pnpm contracts:generate`.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
