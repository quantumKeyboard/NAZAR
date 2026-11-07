import {
  PIPELINE_STAGES,
  type Job,
  type StageName,
  type StageRunState,
  type StageState,
} from '@nazar/contracts';
import { useNow } from '../api/useNow';
import { gb } from '../lib/contributions';
import { formatDuration, stageElapsedS, stageFraction } from '../lib/estimate';
import { isTerminal } from '../lib/jobState';
import { Card, Tag } from './ui';

export const STAGE_COPY: Record<
  StageName,
  { label: string; detail: string; model: string; optional: boolean }
> = {
  preprocessing: {
    label: 'Sample frames',
    detail: 'Cut the video into time windows and keep the sharpest frame of each',
    model: 'FFmpeg + OpenCV',
    optional: false,
  },
  dynamic_analysis: {
    label: 'Moving objects',
    detail: 'Find people and vehicles, mask the ones that move so they cannot blur the scene',
    model: 'YOLO11 + SAM 2',
    optional: true,
  },
  geometry: {
    label: 'Camera poses and points',
    detail: 'Where every camera was, and a dense 3D point for every confident pixel',
    model: 'Fast3R',
    optional: false,
  },
  depth_priors: {
    label: 'Depth and surface direction',
    detail: 'Per-pixel depth fills holes in the points; normals tell the mesh which way is out',
    model: 'Metric3D-v2',
    optional: true,
  },
  sensor_optimization: {
    label: 'GPS alignment',
    detail: 'Only with telemetry: scales and places the model in metres',
    model: 'Sim(3) fit',
    optional: true,
  },
  dense_reconstruction: {
    label: 'Fuse and splat',
    detail: 'Merge the points, then train a photoreal Gaussian splat against your frames',
    model: 'Fusion + gsplat',
    optional: false,
  },
  surface_processing: {
    label: 'Solid surface',
    detail:
      'Fuse the points into a smooth, closed model and paint it from your full-resolution frames; what nobody saw is labelled inferred',
    model: 'Height map + texture',
    optional: true,
  },
  analysis: {
    label: 'Confidence',
    detail: 'Observed, estimated or inferred for every point, and what each model contributed',
    model: 'Report',
    optional: false,
  },
  exporting: {
    label: 'Export and report',
    detail: 'Mesh, splat, point cloud (PLY, LAS, GLB), camera path and the report',
    model: 'Writers',
    optional: false,
  },
};

const STATE_TEXT: Record<StageRunState, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Done',
  skipped: 'Skipped',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function progressText(state: StageState): string | null {
  const p = state.progress;
  if (!p) return null;
  if (p.total != null && p.completed != null) {
    const unit = p.unit ? ` ${p.unit}` : '';
    return `${p.message} · ${p.completed.toLocaleString()} / ${p.total.toLocaleString()}${unit}`;
  }
  return p.message;
}

/** Pipeline stages, live when a job exists. Skips show their reason; nothing is invented. */
export function PipelinePlan({ job }: { job: Job | null }) {
  const states = new Map(job?.stages.map((s) => [s.stage, s]));
  const now = useNow(job !== null && !isTerminal(job));
  return (
    <Card aria-labelledby="pipeline-heading" data-testid="pipeline">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="pipeline-heading" className="mono-label text-slate">
          Reconstruction pipeline
        </h2>
        <Tag tone="mist">{job ? `Attempt ${String(job.attempt)}` : 'No job yet'}</Tag>
      </div>
      <ol className="mt-4 grid gap-px overflow-hidden rounded-card-sm bg-mist sm:grid-cols-3">
        {PIPELINE_STAGES.map((stage, index) => {
          const copy = STAGE_COPY[stage];
          const state = states.get(stage);
          const run = state?.state ?? 'pending';
          const detail = state ? (state.reason ?? progressText(state)) : null;
          const planned = job?.estimate?.stages[stage];
          const fraction = run === 'running' ? stageFraction(state) : null;
          const elapsed = run === 'running' ? stageElapsedS(state, now) : 0;
          return (
            <li
              key={stage}
              className={`flex flex-col gap-1 p-4 transition-colors duration-300 motion-reduce:transition-none ${
                run === 'running' ? 'bg-mint' : 'bg-paper'
              }`}
              data-stage={stage}
              data-state={run}
            >
              <span className="mono-label flex justify-between gap-2 text-smoke">
                <span>
                  {String(index + 1).padStart(2, '0')}
                  {copy.optional ? ' · optional' : ''}
                </span>
                {job && (
                  <span className={run === 'pending' ? '' : 'text-carbon'}>{STATE_TEXT[run]}</span>
                )}
              </span>
              <span className={`font-medium ${run === 'skipped' ? 'text-slate' : ''}`}>
                {copy.label}
              </span>
              <span className="mono-label text-smoke">{copy.model}</span>
              <span className="text-sm text-slate">{detail ?? copy.detail}</span>
              {fraction !== null && (
                <span
                  className="mt-1 block h-1.5 overflow-hidden rounded-tag bg-paper"
                  role="progressbar"
                  aria-label={`${copy.label} progress`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(fraction * 100)}
                >
                  <span
                    className="block h-full bg-carbon transition-[width] duration-500 motion-reduce:transition-none"
                    style={{ width: `${String(Math.round(fraction * 100))}%` }}
                  />
                </span>
              )}
              <span className="font-mono text-xs text-slate" data-testid={`timing-${stage}`}>
                {run === 'completed' && state?.duration_s != null
                  ? `${state.duration_s.toFixed(1)} s${planned ? ` (forecast ${formatDuration(planned)})` : ''}`
                  : run === 'running'
                    ? `${formatDuration(elapsed)} so far${planned ? ` · forecast ${formatDuration(planned)}` : ''}`
                    : run === 'pending' && planned
                      ? `forecast ${formatDuration(planned)}`
                      : ''}
              </span>
              {state?.peak_vram_mib != null && (
                <span className="font-mono text-xs text-slate" data-testid={`vram-${stage}`}>
                  GPU memory {gb(state.peak_vram_mib)}
                  {state.budget_mib ? ` of ${gb(state.budget_mib)} budget` : ''}
                </span>
              )}
            </li>
          );
        })}
      </ol>
      {!job && (
        <p className="mt-4 text-sm text-slate">
          Stages are listed for orientation. Progress appears only when a job reports it.
        </p>
      )}
    </Card>
  );
}
