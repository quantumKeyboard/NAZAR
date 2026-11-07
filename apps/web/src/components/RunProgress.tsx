import type { Job } from '@nazar/contracts';
import { useNow } from '../api/useNow';
import {
  forecast,
  formatClock,
  formatDuration,
  formatRange,
  stageFraction,
  stagePosition,
} from '../lib/estimate';
import { isTerminal } from '../lib/jobState';
import { formatFps } from '../lib/sampling';
import { STAGE_COPY, progressText } from './PipelinePlan';

/** Where the job is, in words and numbers: step, model, counts, elapsed and time left. */
export function RunProgress({ job }: { job: Job }) {
  const active = !isTerminal(job);
  const now = useNow(active);
  const position = stagePosition(job);
  const running = job.stages.find((s) => s.state === 'running');
  const f = active ? forecast(job, now) : null;
  const sampling = job.sampling;

  if (!active) {
    const total = job.stages.reduce((sum, s) => sum + (s.duration_s ?? 0), 0);
    return (
      <p className="mt-2 text-sm text-slate" data-testid="run-summary">
        {sampling
          ? `${String(sampling.frames)} frames at ${formatFps(sampling.fps_effective)}. `
          : ''}
        {job.status === 'completed' ? `Finished in ${formatDuration(total)}.` : ''}
      </p>
    );
  }

  const stageFrac = stageFraction(running);
  const detail = running ? progressText(running) : null;
  const pct = f ? Math.round(f.fraction * 100) : null;

  return (
    <div className="mt-4" data-testid="run-progress">
      <p className="text-sm text-slate" data-testid="run-step">
        {position && running ? (
          <>
            Step {position.index} of {position.count} ·{' '}
            <strong className="text-carbon">{STAGE_COPY[position.stage].label}</strong> ·{' '}
            {STAGE_COPY[position.stage].model}
          </>
        ) : (
          'Waiting to start'
        )}
        {stageFrac !== null && running
          ? ` · ${String(Math.round(stageFrac * 100))} % of this step`
          : ''}
      </p>
      {detail && <p className="mt-1 text-sm text-slate">{detail}</p>}

      {pct !== null && f && (
        <>
          <div
            className="mt-3 h-2.5 overflow-hidden rounded-tag bg-mist"
            role="progressbar"
            aria-label="Overall progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
          >
            <div
              className="h-full bg-carbon transition-[width] duration-700 motion-reduce:transition-none"
              style={{ width: `${String(pct)}%` }}
            />
          </div>
          <p className="mt-2 flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-slate">
            <span data-testid="run-elapsed">Elapsed {formatClock(f.elapsedS)}</span>
            <span data-testid="run-remaining">{formatRange(f.lowS, f.highS)} left</span>
            <span>
              {pct} % · forecast total {formatDuration(job.estimate?.total_s ?? f.totalS)}
            </span>
          </p>
          {job.estimate && (
            <p className="mt-1 text-xs text-slate" data-testid="run-basis">
              {job.estimate.basis}
              {f.pace > 1.25
                ? ' This run is slower than forecast, so the time left was raised.'
                : ''}
              {f.pace < 0.8 ? ' This run is faster than forecast.' : ''}
            </p>
          )}
        </>
      )}
      {!f && <p className="mt-2 text-xs text-slate">No time forecast is available for this run.</p>}
      {sampling && (
        <p className="mt-2 text-xs text-slate" data-testid="run-sampling">
          Using {sampling.frames} frames ({formatFps(sampling.fps_effective)}
          {sampling.auto ? ', chosen automatically from the video length' : ''}
          {sampling.clamped
            ? `, lowered from the ${formatFps(sampling.fps_requested)} wanted by the ${String(sampling.cap)}-frame limit`
            : ''}
          ).
        </p>
      )}
    </div>
  );
}
