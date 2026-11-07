import type { ComputeTarget, Job, PipelineMode } from '@nazar/contracts';
import { useCallback, useEffect, useRef, useState } from 'react';
import { applyEvent, isTerminal } from '../lib/jobState';
import { browserSocket, JobStream, streamUrl, type StreamState } from '../lib/jobStream';
import { errorMessage } from './client';
import { jobs as api } from './jobs';

/**
 * The mission's latest job, kept current by the live event stream. The REST snapshot is the
 * starting point; events are applied strictly in `seq` order and a gap triggers a reload.
 */
export function useJob(missionId: string | null) {
  const [job, setJobState] = useState<Job | null>(null);
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const [stream, setStream] = useState<StreamState>('closed');
  const [error, setError] = useState<string | null>(null);
  const current = useRef<Job | null>(null);

  const setJob = useCallback((next: Job | null) => {
    current.current = next;
    setJobState(next);
  }, []);

  useEffect(() => {
    if (!missionId) return;
    const controller = new AbortController();
    api
      .list(missionId, controller.signal)
      .then((rows) => {
        setJob(rows[0] ?? null);
        setLoadedFor(missionId);
        setError(null);
      })
      .catch(() => {
        if (!controller.signal.aborted) setError('Jobs could not be loaded.');
      });
    return () => {
      controller.abort();
    };
  }, [missionId, setJob]);

  const jobId = job?.id ?? null;
  const active = job !== null && !isTerminal(job);

  useEffect(() => {
    if (!jobId || !active) return;
    const reload = () => {
      api
        .get(jobId)
        .then(setJob)
        .catch(() => undefined);
    };
    const s = new JobStream(
      (after) => streamUrl(jobId, after),
      current.current?.last_seq ?? 0,
      (event) => {
        const base = current.current;
        if (base?.id !== jobId) return;
        const result = applyEvent(base, event);
        if (result.kind === 'applied') setJob(result.job);
        else if (result.kind === 'gap') reload();
      },
      setStream,
      {
        openSocket: browserSocket,
        fetchEvents: (after) => api.events(jobId, after),
        schedule: (fn, ms) => {
          const t = setTimeout(fn, ms);
          return () => {
            clearTimeout(t);
          };
        },
      },
    );
    s.start();
    return () => {
      s.stop();
    };
  }, [jobId, active, setJob]);

  const run = useCallback(
    async (action: () => Promise<Job>) => {
      try {
        setJob(await action());
        setError(null);
      } catch (err) {
        setError(errorMessage((err as { body?: unknown }).body, 'The request failed.'));
      }
    },
    [setJob],
  );

  const visible = missionId && loadedFor === missionId ? job : null;
  return {
    job: visible,
    loaded: missionId !== null && loadedFor === missionId,
    stream: active ? stream : 'closed',
    error,
    // `fps` null or undefined = automatic frame count (the server chooses from the length).
    start: (pipeline: PipelineMode, compute: ComputeTarget = 'local', fps?: number | null) =>
      missionId
        ? run(() => api.create(missionId, pipeline, compute, fps ?? undefined))
        : Promise.resolve(),
    cancel: () => (jobId ? run(() => api.cancel(jobId)) : Promise.resolve()),
    retry: () => (jobId ? run(() => api.retry(jobId)) : Promise.resolve()),
  };
}
