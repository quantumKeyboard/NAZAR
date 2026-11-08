import { PIPELINE_STAGES, type Job, type StageName, type StageState } from '@nazar/contracts';

/**
 * Time remaining for a running job, from the worker's per-stage forecast and what has really
 * happened so far. Nothing here is invented: with no forecast there is no number.
 *
 *  - a finished stage counts its real duration; a skipped stage counts nothing;
 *  - the pace (real time / forecast time of the finished stages) rescales what is still to
 *    come, so a slow machine's estimate grows as soon as it shows;
 *  - the running stage uses its own progress when the worker reports counts (frames,
 *    iterations), otherwise the forecast minus the time already spent, never below 10 %.
 */
export interface Forecast {
  elapsedS: number;
  remainingS: number;
  totalS: number;
  /** 0-1, by time (not by number of stages). */
  fraction: number;
  /** Plausible range for the remaining time. */
  lowS: number;
  highS: number;
  /** How far ahead of or behind the forecast the job is: 1 = on time. */
  pace: number;
}

const PACE_MIN = 0.4;
const PACE_MAX = 4;

function clampPace(pace: number): number {
  return Math.min(PACE_MAX, Math.max(PACE_MIN, pace));
}

export function stageElapsedS(state: StageState | undefined, nowMs: number): number {
  if (state?.state !== 'running' || !state.started_at) return 0;
  return Math.max(0, (nowMs - Date.parse(state.started_at)) / 1000);
}

/**
 * Share of the fuse-and-splat stage taken by fusing (the rest is splat training). Measured
 * on the RTX 4050 run: fusion ~45 s of ~340 s. Its progress counts something else than the
 * splat's iterations, so without this the bar would reach 98 % and then fall back to 6 %.
 */
const FUSION_SHARE = 0.12;

export function stageFraction(state: StageState | undefined): number | null {
  const p = state?.progress;
  if (p?.total != null && p.total > 0 && p.completed != null) {
    const done = Math.min(1, p.completed / p.total);
    if (state?.stage === 'dense_reconstruction') {
      return p.unit === 'iterations'
        ? FUSION_SHARE + done * (1 - FUSION_SHARE)
        : done * FUSION_SHARE;
    }
    return done;
  }
  return null;
}

export function forecast(job: Job, nowMs: number): Forecast | null {
  const est = job.estimate;
  if (!est) return null;
  const planned = (stage: StageName) => est.stages[stage] ?? 0;
  const byStage = new Map(job.stages.map((s) => [s.stage, s]));

  let plannedDone = 0;
  let actualDone = 0;
  for (const stage of PIPELINE_STAGES) {
    const s = byStage.get(stage);
    if (s?.state === 'completed' && s.duration_s != null && planned(stage) > 0) {
      plannedDone += planned(stage);
      actualDone += s.duration_s;
    }
  }
  const pace = plannedDone > 0 ? clampPace(actualDone / plannedDone) : 1;

  let doneS = 0;
  let remainingS = 0;
  for (const stage of PIPELINE_STAGES) {
    const s = byStage.get(stage);
    const forecastS = planned(stage) * pace;
    switch (s?.state) {
      case 'completed':
        doneS += s.duration_s ?? 0;
        break;
      case 'skipped':
      case 'failed':
      case 'cancelled':
        break;
      case 'running': {
        const spent = stageElapsedS(s, nowMs);
        const f = stageFraction(s);
        doneS += spent;
        if (f !== null && f >= 0.1) {
          remainingS += Math.max(0, spent / f - spent);
        } else {
          remainingS += Math.max(forecastS - spent, 0.1 * forecastS);
        }
        break;
      }
      default:
        remainingS += forecastS;
    }
  }
  const totalS = doneS + remainingS;
  return {
    elapsedS: doneS,
    remainingS,
    totalS,
    fraction: totalS > 0 ? Math.min(1, doneS / totalS) : 0,
    lowS: remainingS * 0.85,
    highS: remainingS * 1.3,
    pace,
  };
}

/** "45 s", "12 min", "1 h 05 min": coarse on purpose, an estimate is not a stopwatch. */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${String(s)} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${String(minutes)} min`;
  const h = Math.floor(minutes / 60);
  return `${String(h)} h ${String(minutes % 60).padStart(2, '0')} min`;
}

/** "about 11-15 min", or "about 12 min" when the range rounds to one value. */
export function formatRange(lowS: number, highS: number): string {
  const lo = formatDuration(lowS);
  const hi = formatDuration(highS);
  if (lo === hi) return `about ${lo}`;
  const [loNum, loUnit] = lo.split(' ');
  const [hiNum, hiUnit] = hi.split(' ');
  if (loUnit === hiUnit && loNum && hiNum && !hi.includes(' h ')) {
    return `about ${loNum}-${hiNum} ${hiUnit ?? ''}`.trim();
  }
  return `about ${lo} to ${hi}`;
}

/** Elapsed mm:ss for the live clock. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${String(m)}:${String(s % 60).padStart(2, '0')}`;
}

/** The stage the job is on, its 1-based position among the stages that will run, and how many. */
export function stagePosition(job: Job): { index: number; count: number; stage: StageName } | null {
  const running = job.stages.find((s) => s.state === 'running');
  if (!running) return null;
  const est = job.estimate;
  const counted = job.stages.filter(
    (s) => s.state !== 'skipped' && (est ? (est.stages[s.stage] ?? 0) > 0 || s === running : true),
  );
  return {
    index: Math.max(1, counted.findIndex((s) => s.stage === running.stage) + 1),
    count: counted.length,
    stage: running.stage,
  };
}
