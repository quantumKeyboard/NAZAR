import { PIPELINE_STAGES, type Job, type StageName, type StageState } from '@nazar/contracts';
import { describe, expect, it } from 'vitest';
import {
  forecast,
  formatClock,
  formatDuration,
  formatRange,
  stageFraction,
  stagePosition,
} from './estimate';

const T0 = Date.parse('2026-11-05T10:00:00Z');
const at = (s: number) => new Date(T0 + s * 1000).toISOString();

const PLAN: Partial<Record<StageName, number>> = {
  preprocessing: 1,
  dynamic_analysis: 6,
  geometry: 26,
  depth_priors: 54,
  sensor_optimization: 0,
  dense_reconstruction: 290,
  surface_processing: 22,
  analysis: 0.2,
  exporting: 1,
};
const TOTAL = Object.values(PLAN).reduce((a, b) => a + b, 0);

function job(stages: Partial<Record<StageName, Partial<StageState>>>): Job {
  return {
    id: 'j',
    mission_id: 'm',
    status: 'geometry',
    pipeline: 'standard',
    attempt: 1,
    last_seq: 9,
    artifacts: [],
    warnings: [],
    created_at: at(0),
    updated_at: at(0),
    estimate: { frames: 22, basis: 'test', stages: PLAN, total_s: TOTAL },
    stages: PIPELINE_STAGES.map((stage) => ({
      stage,
      state: 'pending' as const,
      ...stages[stage],
    })),
  };
}

describe('forecast', () => {
  it('has no number without a worker forecast', () => {
    const j = { ...job({}), estimate: null };
    expect(forecast(j, T0)).toBeNull();
  });

  it('before anything runs, the remaining time is the whole forecast', () => {
    const f = forecast(job({}), T0);
    expect(f?.remainingS).toBeCloseTo(TOTAL);
    expect(f?.fraction).toBe(0);
  });

  it('finished stages count their real time and skipped stages nothing', () => {
    const f = forecast(
      job({
        preprocessing: { state: 'completed', duration_s: 1 },
        dynamic_analysis: { state: 'skipped', reason: 'no weights' },
        geometry: { state: 'running', started_at: at(1) },
      }),
      T0 + 11_000,
    );
    expect(f?.elapsedS).toBeCloseTo(1 + 10);
    // pace 1 (preprocessing on time): geometry 26 - 10 spent + the later stages.
    expect(f?.remainingS).toBeCloseTo(16 + 54 + 290 + 22 + 0.2 + 1, 0);
  });

  it('a slow machine stretches what is still to come', () => {
    const slow = forecast(
      job({
        preprocessing: { state: 'completed', duration_s: 2 },
        dynamic_analysis: { state: 'completed', duration_s: 12 },
      }),
      T0 + 14_000,
    );
    expect(slow?.pace).toBeCloseTo(2, 1);
    expect(slow?.remainingS).toBeCloseTo((26 + 54 + 290 + 22 + 0.2 + 1) * 2, 0);
  });

  it('uses the running stage progress once it is meaningful', () => {
    const f = forecast(
      job({
        dense_reconstruction: {
          state: 'running',
          started_at: at(0),
          progress: { message: 'splat', completed: 1000, total: 4000, unit: 'iterations' },
        },
      }),
      T0 + 100_000,
    );
    // 100 s for the first 34 % of the stage (fusion + a quarter of the iterations).
    const denseLeft = 100 / 0.34 - 100;
    expect(f?.remainingS).toBeGreaterThan(denseLeft + 22);
    expect(f?.remainingS).toBeLessThan(denseLeft + 22 + 5 + 120 + 80);
  });

  it('never reports zero while a stage is still running', () => {
    const f = forecast(
      job({ geometry: { state: 'running', started_at: at(0) } }),
      T0 + 500_000, // far past the forecast
    );
    expect(f?.remainingS).toBeGreaterThan(0);
  });

  it('gives a range around the remaining time', () => {
    const f = forecast(job({}), T0);
    expect(f?.lowS).toBeLessThan(f?.remainingS ?? 0);
    expect(f?.highS).toBeGreaterThan(f?.remainingS ?? 0);
  });
});

describe('formatting', () => {
  it('rounds coarsely', () => {
    expect(formatDuration(0.4)).toBe('0 s');
    expect(formatDuration(45)).toBe('45 s');
    expect(formatDuration(11 * 60 + 20)).toBe('11 min');
    expect(formatDuration(65 * 60)).toBe('1 h 05 min');
  });
  it('writes a range', () => {
    expect(formatRange(11 * 60, 15 * 60)).toBe('about 11-15 min');
    expect(formatRange(100, 104)).toBe('about 2 min');
    expect(formatRange(40, 90)).toBe('about 40 s to 2 min');
  });
  it('clock', () => {
    expect(formatClock(0)).toBe('0:00');
    expect(formatClock(370.9)).toBe('6:10');
  });
});

describe('stagePosition', () => {
  it('counts only the stages that will run', () => {
    const j = job({
      preprocessing: { state: 'completed', duration_s: 1 },
      dynamic_analysis: { state: 'skipped' },
      geometry: { state: 'running', started_at: at(0) },
    });
    // sensor_optimization (0 s planned) and the skipped stage are not counted.
    expect(stagePosition(j)).toEqual({ index: 2, count: 7, stage: 'geometry' });
  });
  it('is null when nothing runs', () => {
    expect(stagePosition(job({}))).toBeNull();
  });
});

describe('stageFraction for the fuse-and-splat stage', () => {
  const dense = (completed: number, total: number, unit: string): StageState => ({
    stage: 'dense_reconstruction',
    state: 'running',
    progress: { message: 'm', completed, total, unit },
  });
  it('fusion progress stays in the first share, splat iterations fill the rest', () => {
    expect(stageFraction(dense(100, 100, 'steps'))).toBeCloseTo(0.12);
    expect(stageFraction(dense(0, 4000, 'iterations'))).toBeCloseTo(0.12);
    expect(stageFraction(dense(2000, 4000, 'iterations'))).toBeCloseTo(0.56);
    expect(stageFraction(dense(4000, 4000, 'iterations'))).toBe(1);
  });
  it('other stages are unchanged', () => {
    expect(
      stageFraction({
        stage: 'geometry',
        state: 'running',
        progress: { message: 'm', completed: 5, total: 10, unit: 'keyframes' },
      }),
    ).toBe(0.5);
  });
});
