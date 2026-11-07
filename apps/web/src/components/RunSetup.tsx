import {
  FPS,
  FRAME_LIMITS,
  type ComputeTarget,
  type ModelReadiness,
  type PreflightReport,
  type WorkerPresence,
} from '@nazar/contracts';
import { useState } from 'react';
import { COMPUTE_LABEL } from '../lib/compute';
import { formatRange } from '../lib/estimate';
import { MANUAL_DEFAULT } from '../lib/fpsPreference';
import {
  FPS_PRESETS,
  clampFps,
  formatFps,
  parseFps,
  planFrames,
  preRunSeconds,
} from '../lib/sampling';
import { Tag } from './ui';

interface Props {
  compute: ComputeTarget;
  presence: WorkerPresence | undefined;
  report: PreflightReport | null;
  /** Frames per second, or null for automatic (the frame count follows the video length). */
  fps: number | null;
  onFps: (fps: number | null) => void;
}

function Model({ m }: { m: ModelReadiness }) {
  return (
    <li title={m.detail} data-testid={`model-${m.name}`} data-ready={m.ready}>
      <Tag tone={m.ready ? 'mint' : 'mist'}>
        <span aria-hidden="true">{m.ready ? '✓' : '✗'}</span>
        {m.name}
        <span className="sr-only">{m.ready ? ' ready' : ' not available'}</span>
      </Tag>
    </li>
  );
}

const MODE_BUTTON = 'rounded-button-sm border-[1.5px] px-3 py-1.5 text-sm';
const on = 'border-carbon bg-carbon text-paper';
const off = 'border-slate text-slate hover:bg-mist';

/** Choose how many frames to use and see what it means before anything runs: how many frames,
 *  how long it should take on this worker, and which models are ready. */
export function RunSetup({ compute, presence, report, fps, onFps }: Props) {
  const [manual, setManual] = useState(fps ?? MANUAL_DEFAULT);
  const [text, setText] = useState(String(fps ?? MANUAL_DEFAULT));
  const auto = fps === null;
  const duration = report?.probe.duration_s ?? null;
  const plan = duration ? planFrames(duration, fps, compute) : null;
  const seconds = plan ? preRunSeconds(presence?.time_model, plan.frames) : null;
  const models = presence?.models ?? null;
  const missing = models?.filter((m) => !m.ready) ?? [];

  const commit = (value: number) => {
    const next = clampFps(value);
    setManual(next);
    setText(String(next));
    onFps(next);
  };

  return (
    <fieldset className="mt-6" data-testid="run-setup">
      <legend className="mono-label text-slate">Frames to use</legend>
      <div className="mt-2 flex flex-wrap items-center gap-2" role="group" aria-label="Frame mode">
        <button
          type="button"
          aria-pressed={auto}
          onClick={() => {
            onFps(null);
          }}
          className={`${MODE_BUTTON} ${auto ? on : off}`}
          data-testid="fps-auto"
        >
          Automatic (recommended)
        </button>
        <button
          type="button"
          aria-pressed={!auto}
          onClick={() => {
            commit(manual);
          }}
          className={`${MODE_BUTTON} ${auto ? off : on}`}
          data-testid="fps-manual"
        >
          Choose frames per second
        </button>
      </div>

      {!auto && (
        <div className="mt-3 flex flex-wrap items-center gap-4">
          <input
            type="range"
            aria-label="Frames per second (slider)"
            min={FPS.min}
            max={FPS.max}
            step={0.25}
            value={fps}
            onChange={(e) => {
              commit(Number(e.target.value));
            }}
            className="w-40 accent-carbon"
            data-testid="fps-slider"
          />
          <label className="flex items-center gap-2 text-sm text-slate">
            <input
              type="text"
              inputMode="decimal"
              aria-label="Frames per second"
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                const parsed = parseFps(e.target.value);
                if (parsed !== null) {
                  setManual(parsed);
                  onFps(parsed);
                }
              }}
              onBlur={() => {
                commit(parseFps(text) ?? manual);
              }}
              className="w-20 rounded-button-sm border-[1.5px] border-slate px-3 py-2 text-carbon"
              data-testid="fps-input"
            />
            fps
          </label>
          <div className="flex gap-2">
            {FPS_PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => {
                  commit(p);
                }}
                aria-pressed={fps === p}
                className={`${MODE_BUTTON} ${fps === p ? on : off}`}
              >
                {formatFps(p)}
              </button>
            ))}
          </div>
        </div>
      )}

      <p className="mt-3 text-sm" data-testid="fps-summary">
        {plan && duration ? (
          plan.tooFew ? (
            <>
              At {formatFps(fps ?? 0)} this {duration.toFixed(0)} s video gives only {plan.frames}{' '}
              frame{plan.frames === 1 ? '' : 's'}; at least {FRAME_LIMITS.min} are needed. Choose a
              higher rate.
            </>
          ) : (
            <>
              <strong>{plan.frames} frames</strong> from a {duration.toFixed(1)} s video (
              {plan.auto
                ? `automatic: about one per second, at least ${String(FRAME_LIMITS.auto_min)}; that is ${formatFps(plan.effectiveFps)}`
                : `about ${formatFps(plan.effectiveFps)}`}
              ).
              {plan.clamped
                ? ` Lowered from ${String(plan.wanted)} to the ${COMPUTE_LABEL[compute]} limit of ${String(plan.cap)}.`
                : ''}{' '}
              Each frame is the sharpest usable one in its stretch of the video.
            </>
          )
        ) : (
          'Reading the video length…'
        )}
      </p>
      <p className="mt-1 text-sm text-slate" data-testid="fps-estimate">
        {seconds !== null && presence?.time_model
          ? `Estimated time: ${formatRange(seconds * 0.85, seconds * 1.3)}. ${presence.time_model.basis}`
          : 'No time estimate until the worker has connected.'}
      </p>

      {models && (
        <div className="mt-4" data-testid="model-readiness">
          <span className="mono-label text-slate">Models on this worker</span>
          <ul className="mt-2 flex flex-wrap gap-2">
            {models.map((m) => (
              <Model key={m.name} m={m} />
            ))}
          </ul>
          {missing.length > 0 && (
            <p className="mt-2 text-sm text-slate">
              {missing.map((m) => `${m.name}: ${m.detail}`).join(' ')}{' '}
              {compute === 'local'
                ? 'On the local GPU a missing model is skipped, with the reason in the report.'
                : 'The college profile needs every model.'}
            </p>
          )}
        </div>
      )}
    </fieldset>
  );
}
