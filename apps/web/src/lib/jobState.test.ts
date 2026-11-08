import type { ArtifactManifest, Job, JobEvent } from '@nazar/contracts';
import { describe, expect, it } from 'vitest';
import { applyEvent, displayCloud, pendingStages } from './jobState';

const TS = '2026-10-27T00:00:00Z';

function job(overrides: Partial<Job> = {}): Job {
  return {
    id: 'j',
    mission_id: 'm',
    status: 'queued',
    pipeline: 'test',
    attempt: 1,
    last_seq: 2,
    stages: pendingStages(),
    artifacts: [],
    warnings: [],
    created_at: TS,
    updated_at: TS,
    ...overrides,
  };
}

const base = { job_id: 'j', ts: TS };

function manifest(kind: ArtifactManifest['kind'], revision: number): ArtifactManifest {
  return {
    id: `${kind}-${String(revision)}`,
    job_id: 'j',
    kind,
    revision,
    media_type: 'application/vnd.nazar.ply',
    byte_size: 10,
    checksum_sha256: 'a'.repeat(64),
    provenance: { producer: 't', test_pipeline: true },
    created_at: TS,
  };
}

function apply(j: Job, event: JobEvent): Job {
  const result = applyEvent(j, event);
  if (result.kind !== 'applied') throw new Error(result.kind);
  return result.job;
}

describe('applyEvent', () => {
  it('walks a stage through started, progress and completed', () => {
    let j = apply(job(), {
      ...base,
      seq: 3,
      status: 'preprocessing',
      stage: 'preprocessing',
      type: 'stage.started',
    });
    j = apply(j, {
      ...base,
      seq: 4,
      status: 'preprocessing',
      stage: 'preprocessing',
      type: 'stage.progress',
      data: { message: 'm', completed: 3, total: 9, unit: 'keyframes' },
    });
    expect(j.stages[0]).toMatchObject({ state: 'running', progress: { completed: 3 } });
    j = apply(j, {
      ...base,
      seq: 5,
      status: 'preprocessing',
      stage: 'preprocessing',
      type: 'stage.completed',
      data: { duration_s: 1.5 },
    });
    expect(j.stages[0]).toMatchObject({
      stage: 'preprocessing',
      state: 'completed',
      duration_s: 1.5,
      started_at: TS,
    });
    expect(j.last_seq).toBe(5);
  });

  it('ignores duplicates and reports gaps instead of guessing', () => {
    const start = job();
    const dup: JobEvent = { ...base, seq: 2, status: 'queued', type: 'cancel.requested' };
    expect(applyEvent(start, dup).kind).toBe('stale');
    const ahead: JobEvent = { ...base, seq: 9, status: 'queued', type: 'cancel.requested' };
    expect(applyEvent(start, ahead).kind).toBe('gap');
  });

  it('keeps the newest artifact revision per kind', () => {
    let j = apply(job(), {
      ...base,
      seq: 3,
      status: 'geometry',
      type: 'artifact.available',
      data: manifest('point_cloud_preview', 2),
    });
    j = apply(j, {
      ...base,
      seq: 4,
      status: 'geometry',
      type: 'artifact.available',
      data: manifest('point_cloud_preview', 1),
    });
    expect(j.artifacts.map((a) => a.revision)).toEqual([2]);
    expect(displayCloud(j)?.id).toBe('point_cloud_preview-2');
    j = apply(j, {
      ...base,
      seq: 5,
      status: 'dense_reconstruction',
      type: 'artifact.available',
      data: manifest('point_cloud', 1),
    });
    expect(displayCloud(j)?.kind).toBe('point_cloud');
  });

  it('keeps the worker estimate, stage start time and GPU peak', () => {
    let j = apply(job(), {
      ...base,
      seq: 3,
      status: 'queued',
      type: 'job.estimate',
      data: { frames: 22, basis: 'test', stages: { geometry: 26 }, total_s: 26 },
    });
    expect(j.estimate?.total_s).toBe(26);
    j = apply(j, {
      ...base,
      seq: 4,
      status: 'geometry',
      stage: 'geometry',
      type: 'stage.started',
    });
    j = apply(j, {
      ...base,
      seq: 5,
      status: 'geometry',
      stage: 'geometry',
      type: 'stage.completed',
      data: { duration_s: 25, peak_vram_mib: 4769, budget_mib: 5632 },
    });
    expect(j.stages.find((x) => x.stage === 'geometry')).toMatchObject({
      state: 'completed',
      peak_vram_mib: 4769,
      budget_mib: 5632,
      started_at: TS,
    });
    j = apply(j, {
      ...base,
      seq: 6,
      status: 'queued',
      type: 'job.queued',
      data: { attempt: 2, reason: 'Retry requested.' },
    });
    expect(j.estimate).toBeNull();
  });

  it('marks the running stage failed and a retry starts a clean attempt', () => {
    let j = apply(job(), {
      ...base,
      seq: 3,
      status: 'geometry',
      stage: 'geometry',
      type: 'stage.started',
    });
    j = apply(j, {
      ...base,
      seq: 4,
      status: 'failed',
      type: 'job.failed',
      data: { reason: 'boom', retryable: true },
    });
    expect(j.stages.find((x) => x.stage === 'geometry')).toMatchObject({
      state: 'failed',
      reason: 'boom',
    });
    expect(j).toMatchObject({ status: 'failed', error: 'boom', retryable: true });
    j = apply(j, {
      ...base,
      seq: 5,
      status: 'queued',
      type: 'job.queued',
      data: { attempt: 2, reason: 'Retry requested.' },
    });
    expect(j).toMatchObject({ status: 'queued', attempt: 2, error: null, artifacts: [] });
    expect(j.stages.every((s) => s.state === 'pending')).toBe(true);
  });
});
