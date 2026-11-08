import type { PreflightReport, PreflightWarning } from '@nazar/contracts';
import { useEffect, useState } from 'react';
import { assets } from '../api/missions';
import { QualityChart } from './QualityChart';
import { Card, Tag } from './ui';

const SEVERITY_ORDER = { blocking: 0, warning: 1, info: 2 } as const;
const SEVERITY_TEXT = { blocking: 'Blocking', warning: 'Warning', info: 'Note' } as const;
const REASON_TEXT = {
  start: 'start',
  novelty: 'new view',
  max_gap: 'time gap',
  end: 'end',
  sampled: 'sampled',
};

function fmtFps(value: number | null) {
  return value ? `${value.toFixed(2)} fps` : 'unknown';
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="mono-label text-smoke">{label}</dt>
      <dd className="text-lg font-medium">{value}</dd>
    </div>
  );
}

function Warning({ warning }: { warning: PreflightWarning }) {
  return (
    <li className="flex flex-col gap-1 py-3 sm:flex-row sm:items-baseline sm:gap-4">
      <Tag
        tone={warning.severity === 'blocking' ? 'carbon' : 'mist'}
        className="shrink-0 self-start"
      >
        {SEVERITY_TEXT[warning.severity]}
      </Tag>
      <span className="text-sm">{warning.message}</span>
    </li>
  );
}

/** Results of server-side validation: facts, warnings, quality over time, keyframes. */
export function PreflightPanel({ assetId }: { assetId: string }) {
  const [report, setReport] = useState<PreflightReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    assets
      .preflight(assetId, controller.signal)
      .then(setReport)
      .catch(() => {
        if (!controller.signal.aborted) setError('The preflight report could not be loaded.');
      });
    return () => {
      controller.abort();
    };
  }, [assetId]);

  if (error) return <Card>{error}</Card>;
  if (!report) {
    return (
      <Card aria-busy="true">
        <p className="text-slate">Loading preflight report…</p>
      </Card>
    );
  }

  const { probe, summary, keyframes, analysis } = report;
  const warnings = [...report.warnings].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  );
  const blocking = warnings.some((w) => w.severity === 'blocking');

  return (
    <Card aria-labelledby="preflight-heading" data-testid="preflight-panel">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="preflight-heading" className="heading">
          Preflight
        </h2>
        <Tag tone={blocking ? 'carbon' : 'mint'}>
          {blocking ? 'Not reconstructable' : 'Usable for reconstruction'}
        </Tag>
      </div>
      <p className="mt-2 text-slate">
        Measured on the server by decoding all {analysis.frames_decoded} frames in{' '}
        {analysis.processing_s.toFixed(1)} s.
      </p>

      <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        <Fact label="Duration" value={`${probe.duration_s.toFixed(1)} s`} />
        <Fact label="Resolution" value={`${probe.display_width}×${probe.display_height}`} />
        <Fact
          label="Frame rate"
          value={`${fmtFps(probe.average_fps)}${probe.variable_frame_rate ? ' · VFR' : ''}`}
        />
        <Fact label="Codec" value={`${probe.codec} · ${probe.container.split(',')[0] ?? ''}`} />
        <Fact
          label="Usable frames"
          value={`${summary.usable_frames} / ${analysis.frames_decoded}`}
        />
        <Fact label="Keyframes" value={String(keyframes.length)} />
        <Fact label="Viewpoint change" value={summary.viewpoint_change.toFixed(2)} />
        <Fact label="Telemetry" value={report.telemetry_present ? 'Present' : 'None'} />
      </dl>

      {warnings.length > 0 && (
        <section aria-labelledby="warnings-heading" className="mt-8">
          <h3 id="warnings-heading" className="mono-label text-slate">
            Findings
          </h3>
          <ul className="divide-y divide-mist" data-testid="preflight-warnings">
            {warnings.map((w) => (
              <Warning key={w.code} warning={w} />
            ))}
          </ul>
        </section>
      )}

      <div className="mt-8">
        <QualityChart report={report} />
      </div>

      <section aria-labelledby="keyframes-heading" className="mt-8">
        <h3 id="keyframes-heading" className="mono-label text-slate">
          Keyframes · {keyframes.length} chosen for reconstruction
        </h3>
        <ol
          className="mt-3 flex snap-x gap-3 overflow-x-auto pb-3"
          data-testid="keyframe-strip"
          tabIndex={0}
          aria-label="Keyframe thumbnails, scrollable"
        >
          {keyframes.map((k) => (
            <li key={k.index} className="flex shrink-0 snap-start flex-col gap-1">
              <a
                href={assets.imageUrl(assetId, 'keyframes', k.image)}
                target="_blank"
                rel="noreferrer"
                className="block overflow-hidden rounded-button bg-mist"
              >
                <img
                  src={assets.imageUrl(assetId, 'thumbs', k.image)}
                  alt={`Keyframe ${k.image} at ${k.t.toFixed(2)} s`}
                  loading="lazy"
                  className="h-28 w-auto"
                />
              </a>
              <span className="font-mono text-[11px] text-slate">
                {k.t.toFixed(1)} s · {REASON_TEXT[k.reason]}
              </span>
            </li>
          ))}
        </ol>
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-slate">Show keyframes as a table</summary>
          <table className="mt-2 w-full font-mono text-xs">
            <thead>
              <tr className="text-left text-smoke">
                <th className="py-1 font-normal">#</th>
                <th className="py-1 font-normal">Time</th>
                <th className="py-1 font-normal">Frame</th>
                <th className="py-1 font-normal">Why</th>
                <th className="py-1 text-right font-normal">Quality</th>
              </tr>
            </thead>
            <tbody>
              {keyframes.map((k) => (
                <tr key={k.index} className="border-t border-mist">
                  <td className="py-1">{k.image}</td>
                  <td className="py-1">{k.t.toFixed(2)} s</td>
                  <td className="py-1">{k.index}</td>
                  <td className="py-1">{REASON_TEXT[k.reason]}</td>
                  <td className="py-1 text-right">{k.quality.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </section>
    </Card>
  );
}
