import {
  PIPELINE_STAGES,
  TERMINAL_STATUSES,
  type ArtifactManifest,
  type Job,
  type JobEvent,
  type StageState,
} from '@nazar/contracts';

export const pendingStages = (): StageState[] =>
  PIPELINE_STAGES.map((stage) => ({ stage, state: 'pending' }));

export const isTerminal = (job: Pick<Job, 'status'>) => TERMINAL_STATUSES.includes(job.status);

export type ApplyResult =
  | { kind: 'applied'; job: Job }
  /** Already applied (duplicate delivery or replay overlap): ignore. */
  | { kind: 'stale' }
  /** Events were missed; reload the snapshot from the server. */
  | { kind: 'gap' };

function setStage(job: Job, next: StageState): StageState[] {
  return job.stages.map((s) => (s.stage === next.stage ? next : s));
}

function upsertArtifact(list: ArtifactManifest[], manifest: ArtifactManifest) {
  const current = list.find((a) => a.kind === manifest.kind);
  // Revisions are monotonic per kind: never let a late, older revision win.
  if (current && current.revision >= manifest.revision) return list;
  return [...list.filter((a) => a.kind !== manifest.kind), manifest];
}

/**
 * Apply one event to a job snapshot. Mirrors the server's snapshot builder, so a client that
 * started from `GET /api/jobs/{id}` and applied every later event ends in the same state as a
 * fresh snapshot. Events must arrive in `seq` order; anything else is reported, not guessed.
 */
export function applyEvent(job: Job, event: JobEvent): ApplyResult {
  if (event.seq <= job.last_seq) return { kind: 'stale' };
  if (event.seq !== job.last_seq + 1) return { kind: 'gap' };
  const running = job.stages.find((s) => s.state === 'running')?.stage;
  let next: Job = { ...job, last_seq: event.seq, status: event.status, updated_at: event.ts };

  switch (event.type) {
    case 'job.queued':
      next = {
        ...next,
        attempt: event.data.attempt,
        stages: pendingStages(),
        artifacts: [],
        warnings: [],
        error: null,
        retryable: false,
        estimate: null,
      };
      break;
    case 'job.estimate':
      next = { ...next, estimate: event.data };
      break;
    case 'stage.started':
      if (event.stage)
        next.stages = setStage(next, {
          stage: event.stage,
          state: 'running',
          started_at: event.ts,
        });
      break;
    case 'stage.progress':
      if (event.stage) {
        const stage = event.stage;
        next.stages = next.stages.map((s) =>
          s.stage === stage ? { ...s, progress: event.data } : s,
        );
      }
      break;
    case 'stage.completed':
      if (event.stage) {
        const stage = event.stage;
        next.stages = setStage(next, {
          stage,
          state: 'completed',
          duration_s: event.data.duration_s,
          peak_vram_mib: event.data.peak_vram_mib ?? null,
          budget_mib: event.data.budget_mib ?? null,
          started_at: job.stages.find((s) => s.stage === stage)?.started_at ?? null,
        });
      }
      break;
    case 'stage.skipped':
      if (event.stage)
        next.stages = setStage(next, {
          stage: event.stage,
          state: 'skipped',
          reason: event.data.reason,
        });
      break;
    case 'warning':
      next.warnings = [...next.warnings, event.data];
      break;
    case 'artifact.available':
      next.artifacts = upsertArtifact(next.artifacts, event.data);
      break;
    case 'job.failed':
      next = { ...next, error: event.data.reason, retryable: event.data.retryable };
      if (running)
        next.stages = setStage(next, {
          stage: running,
          state: 'failed',
          reason: event.data.reason,
        });
      break;
    case 'job.cancelled':
      if (running) next.stages = setStage(next, { stage: running, state: 'cancelled' });
      break;
    default:
      break;
  }
  return { kind: 'applied', job: next };
}

const KIND_PRIORITY = ['point_cloud', 'point_cloud_preview'] as const;

/** The point cloud to display: the final cloud once it exists, otherwise the newest preview. */
export function displayCloud(job: Job | null): ArtifactManifest | null {
  if (!job) return null;
  for (const kind of KIND_PRIORITY) {
    const found = job.artifacts.find((a) => a.kind === kind);
    if (found) return found;
  }
  return null;
}

/** The solid surface mesh (GLB), when the job produced one. */
export function displayMesh(job: Job | null): ArtifactManifest | null {
  return job?.artifacts.find((a) => a.kind === 'mesh') ?? null;
}

/** The photoreal Gaussian splat (PLY), when the job produced one. */
export function displaySplat(job: Job | null): ArtifactManifest | null {
  return job?.artifacts.find((a) => a.kind === 'gaussian_scene') ?? null;
}
