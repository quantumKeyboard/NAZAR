import {
  FPS,
  type ArtifactKind,
  type ArtifactManifest,
  type ComputeTarget,
  type Job,
  type PipelineMode,
  type PreflightReport,
  type WorkerPresence,
} from '@nazar/contracts';
import { useEffect, useRef, useState } from 'react';
import { jobs } from '../api/jobs';
import {
  celebrate,
  claimCompletion,
  hapticsSupported,
  loadPrefs,
  savePrefs,
  type CompletionFeedback,
} from '../lib/completion';
import { COMPUTE_LABEL, loadCompute, presenceText, saveCompute } from '../lib/compute';
import { loadFps, saveFps } from '../lib/fpsPreference';
import { planFrames } from '../lib/sampling';
import { isTerminal } from '../lib/jobState';
import { jobMethod, METHOD_TAG } from '../lib/provenance';
import type { StreamState } from '../lib/jobStream';
import { Contributions } from './Contributions';
import { STAGE_COPY } from './PipelinePlan';
import { RunProgress } from './RunProgress';
import { RunSetup } from './RunSetup';
import { Button, Card, Tag } from './ui';

const KIND_TEXT: Record<ArtifactKind, string> = {
  keyframes: 'Keyframes',
  point_cloud_preview: 'Point cloud preview',
  point_cloud: 'Point cloud',
  camera_trajectory: 'Camera path',
  depth_maps: 'Depth maps',
  dynamic_masks: 'Dynamic masks',
  gaussian_scene: 'Photoreal Gaussian splat (PLY)',
  mesh: 'Solid surface mesh (GLB)',
  report: 'Processing report',
  export: 'Export',
};

const STREAM_TEXT: Record<StreamState, string> = {
  connecting: 'Connecting…',
  live: 'Live',
  reconnecting: 'Reconnecting…',
  closed: '',
};

function statusText(job: Job): string {
  switch (job.status) {
    case 'created':
    case 'validating':
    case 'queued':
      return job.compute === 'college'
        ? 'Queued. Waiting for the college GPU worker to pick it up.'
        : 'Queued. Waiting for a worker to pick it up.';
    case 'cancel_requested':
      return 'Cancelling. The worker stops at its next safe point.';
    case 'completed':
      return 'Completed. Every stage finished or was skipped with a reason.';
    case 'failed':
      return `Failed. ${job.error ?? ''}`;
    case 'cancelled':
      return 'Cancelled.';
    default:
      return `Running: ${STAGE_COPY[job.status].label}.`;
  }
}

function fmtBytes(n: number) {
  if (n < 1024) return `${String(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function extension(a: ArtifactManifest) {
  if (a.media_type.includes('ply')) return 'ply';
  if (a.media_type.includes('las')) return 'las';
  if (a.media_type.includes('gltf-binary')) return 'glb';
  return a.media_type.includes('json') ? 'json' : 'bin';
}

const EXPORT_NAMES: Partial<Record<string, string>> = {
  las: 'Point cloud (LAS)',
  glb: 'Point cloud + camera path (GLB)',
};

function outputName(a: ArtifactManifest) {
  return (a.kind === 'export' ? EXPORT_NAMES[extension(a)] : undefined) ?? KIND_TEXT[a.kind];
}

function Inventory({ artifacts }: { artifacts: ArtifactManifest[] }) {
  return (
    <ul className="divide-y divide-mist" data-testid="job-outputs">
      {artifacts.map((a) => (
        <li key={a.id} className="flex flex-wrap items-baseline justify-between gap-2 py-3">
          <span className="flex flex-col">
            <span className="font-medium">{outputName(a)}</span>
            <span className="font-mono text-xs text-slate">
              rev {a.revision} · {fmtBytes(a.byte_size)}
              {a.point_count != null ? ` · ${a.point_count.toLocaleString()} pts` : ''}
              {a.frame ? ` · ${a.frame.units}, ${a.frame.scale_status} scale` : ''}
            </span>
          </span>
          <a
            className="rounded-button-sm border-[1.5px] border-slate px-3 py-2 text-sm text-slate hover:bg-mist"
            href={jobs.artifactUrl(a.id)}
            download={`${a.kind}-r${String(a.revision)}.${extension(a)}`}
          >
            Download
          </a>
        </li>
      ))}
    </ul>
  );
}

interface Props {
  job: Job | null;
  loaded: boolean;
  canStart: boolean;
  stream: StreamState;
  error: string | null;
  /** Presence of the local and college workers (from readiness). */
  workers: WorkerPresence[];
  /** The ready video's preflight report (duration), for the frames preview. */
  report: PreflightReport | null;
  /** `fps` null = automatic frame count. */
  onStart: (pipeline: PipelineMode, compute: ComputeTarget, fps: number | null) => Promise<void>;
  onCancel: () => Promise<void>;
  onRetry: () => Promise<void>;
}

/** Start, follow, cancel and retry a reconstruction job; list its outputs when complete. */
export function JobPanel({
  job,
  loaded,
  canStart,
  stream,
  error,
  workers,
  report,
  onStart,
  onCancel,
  onRetry,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [compute, setCompute] = useState<ComputeTarget>(loadCompute);
  const [fps, setFpsState] = useState(loadFps);
  const setFps = (next: number | null) => {
    setFpsState(next);
    saveFps(next);
  };
  const [prefs, setPrefs] = useState(loadPrefs);
  const [feedback, setFeedback] = useState<CompletionFeedback | null>(null);
  const watched = useRef<string | null>(null);

  // Celebrate only a completion we watched happen, once per job id.
  useEffect(() => {
    if (!job) return;
    if (!isTerminal(job)) {
      watched.current = job.id;
      return;
    }
    if (job.status === 'completed' && watched.current === job.id && claimCompletion(job.id)) {
      watched.current = null;
      // Feedback runs as a side effect of the observed transition, exactly once.

      setFeedback(celebrate(prefs));
    }
  }, [job, prefs]);

  const act = (fn: () => Promise<void>) => () => {
    setBusy(true);
    void fn().finally(() => {
      setBusy(false);
    });
  };

  const togglePref = (key: keyof typeof prefs) => {
    const next = { ...prefs, [key]: !prefs[key] };
    setPrefs(next);
    savePrefs(next);
  };

  const plan = report ? planFrames(report.probe.duration_s, fps, compute) : null;
  const finished = job ? isTerminal(job) : true;
  const showStart = loaded && canStart && finished;
  const outputs = job?.status === 'completed' ? job.artifacts : [];
  const method = job ? jobMethod(job.artifacts) : null;

  return (
    <Card aria-labelledby="job-heading" data-testid="job-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="job-heading" className="mono-label text-slate">
          Reconstruction job
        </h2>
        <div className="flex gap-2">
          {job?.pipeline === 'test' && <Tag tone="carbon">Test pipeline</Tag>}
          {job?.pipeline === 'standard' && method && method !== 'test' && (
            <Tag tone="mint">{METHOD_TAG[method]}</Tag>
          )}
          {job && (
            <Tag tone="mist" data-testid="job-compute">
              {COMPUTE_LABEL[job.compute ?? 'local']}
            </Tag>
          )}
          {job && STREAM_TEXT[stream] && <Tag tone="mist">{STREAM_TEXT[stream]}</Tag>}
        </div>
      </div>

      {job && (
        <p
          className={`mt-4 text-lg font-medium ${job.status === 'completed' ? 'animate-[nazar-pop_420ms_ease-out] motion-reduce:animate-none' : ''}`}
          role="status"
          aria-live="polite"
          data-testid="job-status"
          data-status={job.status}
        >
          {statusText(job)}
        </p>
      )}
      {job && job.pipeline === 'standard' && <RunProgress job={job} />}
      {job?.pipeline === 'test' && (
        <p className="mt-2 text-sm text-slate">
          The test pipeline checks the job, event and artifact plumbing with a fixed synthetic
          scene. Its geometry is not a reconstruction of your video.
        </p>
      )}

      {job && job.warnings.length > 0 && (
        <ul className="mt-4 flex flex-col gap-2" data-testid="job-warnings">
          {job.warnings.map((w) => (
            <li key={w.code} className="flex items-baseline gap-3 text-sm">
              <Tag tone={w.severity === 'critical' ? 'carbon' : 'mist'} className="shrink-0">
                {w.severity}
              </Tag>
              {w.message}
            </li>
          ))}
        </ul>
      )}

      {job?.status === 'completed' && job.pipeline === 'standard' && <Contributions job={job} />}

      {outputs.length > 0 && (
        <div className="mt-6">
          <h3 className="mono-label text-slate">Outputs</h3>
          <Inventory artifacts={outputs} />
          {feedback && (
            <p className="text-xs text-slate" data-testid="completion-feedback">
              {feedback.sound ? 'Completion tone played. ' : ''}
              {/* The browser accepted the request; desktops usually have no vibration motor. */}
              {feedback.haptics ? 'Vibration requested. ' : ''}
            </p>
          )}
        </div>
      )}

      {error && (
        <p role="alert" className="mt-4 text-sm">
          {error}
        </p>
      )}

      {showStart && (
        <fieldset className="mt-6" data-testid="compute-toggle">
          <legend className="mono-label text-slate">Run on</legend>
          <div className="mt-2 inline-flex rounded-button-sm border-[1.5px] border-slate p-0.5">
            {(['local', 'college'] as const).map((target) => (
              <label
                key={target}
                className={`cursor-pointer rounded-button-sm px-3 py-1.5 text-sm has-[:focus-visible]:outline has-[:focus-visible]:outline-2 ${
                  compute === target ? 'bg-carbon text-paper' : 'text-slate hover:bg-mist'
                }`}
              >
                <input
                  type="radio"
                  name="compute"
                  value={target}
                  className="sr-only"
                  checked={compute === target}
                  onChange={() => {
                    setCompute(target);
                    saveCompute(target);
                  }}
                />
                {COMPUTE_LABEL[target]}
              </label>
            ))}
          </div>
          <p className="mt-2 text-sm text-slate" data-testid="compute-presence">
            {presenceText(
              compute,
              workers.find((w) => w.location === compute),
            )}
          </p>
        </fieldset>
      )}

      {showStart && (
        <RunSetup
          compute={compute}
          presence={workers.find((w) => w.location === compute)}
          report={report}
          fps={fps}
          onFps={setFps}
        />
      )}

      <div className="mt-6 flex flex-wrap gap-3">
        {showStart && (
          <>
            <Button
              disabled={busy || (plan?.tooFew ?? false)}
              onClick={act(() => onStart('standard', compute, fps))}
              data-testid="reconstruct"
            >
              Reconstruct
            </Button>
            <Button
              variant="ghost"
              disabled={busy}
              onClick={act(() => onStart('test', compute, FPS.default))}
            >
              Run test pipeline
            </Button>
          </>
        )}
        {job && !finished && job.status !== 'cancel_requested' && (
          <Button variant="ghost" disabled={busy} onClick={act(onCancel)}>
            Cancel job
          </Button>
        )}
        {job && (job.status === 'cancelled' || (job.status === 'failed' && job.retryable)) && (
          <Button disabled={busy} onClick={act(onRetry)}>
            Retry
          </Button>
        )}
      </div>
      {showStart && (
        <p className="mt-3 text-sm text-slate">
          Reconstruct runs every model the worker has: moving-object masks, Fast3R geometry,
          Metric3D depth, a Gaussian splat and a solid mesh, and tells you what each one added. The
          result is in a local frame with unknown scale unless GPS telemetry is attached.
        </p>
      )}
      {!canStart && !job && loaded && (
        <p className="mt-4 text-sm text-slate">Upload a video and wait for validation first.</p>
      )}

      <fieldset className="mt-6 flex flex-wrap gap-4 text-sm text-slate">
        <legend className="sr-only">Completion feedback</legend>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={prefs.sound}
            onChange={() => {
              togglePref('sound');
            }}
          />
          Completion sound
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={prefs.haptics && hapticsSupported()}
            disabled={!hapticsSupported()}
            onChange={() => {
              togglePref('haptics');
            }}
          />
          Vibrate{hapticsSupported() ? '' : ' (not supported here)'}
        </label>
      </fieldset>
    </Card>
  );
}
