import type { Job } from '@nazar/contracts';
import { useEffect, useState } from 'react';
import { request } from '../api/client';
import { describeContributions, gb, type Contribution } from '../lib/contributions';
import { Tag } from './ui';

interface Report {
  contributions?: Record<string, Contribution>;
  resources?: {
    peak_vram_mib?: number;
    gpu_budget_mib?: number;
    within_gpu_budget?: boolean | null;
  };
  limitations?: string[];
}

/** After a job: what each model contributed, with its numbers, from the processing report. */
export function Contributions({ job }: { job: Job }) {
  const reportId = job.artifacts.find((a) => a.kind === 'report')?.id ?? null;
  const [loaded, setLoaded] = useState<{ id: string; report: Report } | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!reportId) return;
    const controller = new AbortController();
    request<Report>('GET', `/api/artifacts/${reportId}/content`, undefined, controller.signal)
      .then((report) => {
        setLoaded({ id: reportId, report });
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      });
    return () => {
      controller.abort();
    };
  }, [reportId]);

  if (!reportId) return null;
  if (error) return <p className="mt-6 text-sm text-slate">The report could not be loaded.</p>;
  const report = loaded?.id === reportId ? loaded.report : null;
  if (!report) return <p className="mt-6 text-sm text-slate">Loading what each model did…</p>;
  const views = describeContributions(report.contributions);
  if (views.length === 0) return null;
  const res = report.resources;

  return (
    <div className="mt-6" data-testid="contributions">
      <h3 className="mono-label text-slate">What each model contributed</h3>
      <ul className="mt-3 flex flex-col gap-3">
        {views.map((v) => (
          <li
            key={v.key}
            className="rounded-card-sm border border-mist p-4"
            data-testid={`contribution-${v.key}`}
            data-applied={v.applied}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-medium">{v.title}</span>
              <span className="flex gap-2">
                <Tag tone="mist">{v.model}</Tag>
                {!v.applied && <Tag tone="carbon">Did not run</Tag>}
                {v.peakVramMib !== null && <Tag tone="mist">GPU {gb(v.peakVramMib)}</Tag>}
              </span>
            </div>
            <p className="mt-1 text-sm">{v.applied ? v.headline : (v.reason ?? v.headline)}</p>
            {v.what && v.applied && <p className="mt-1 text-xs text-slate">{v.what}</p>}
            {v.facts.length > 0 && (
              <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
                {v.facts.map(([label, value]) => (
                  <div key={label} className="flex flex-col">
                    <dt className="text-smoke">{label}</dt>
                    <dd className="font-mono">{value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </li>
        ))}
      </ul>
      {res?.peak_vram_mib != null && res.gpu_budget_mib ? (
        <p className="mt-3 text-sm text-slate" data-testid="vram-summary">
          Peak GPU memory {gb(res.peak_vram_mib)} of the {gb(res.gpu_budget_mib)} budget:{' '}
          {res.within_gpu_budget ? 'within budget.' : 'over budget.'}
        </p>
      ) : null}
      {report.limitations && report.limitations.length > 0 && (
        <ul className="mt-3 list-disc pl-5 text-xs text-slate">
          {report.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
