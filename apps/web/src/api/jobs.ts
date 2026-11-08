import type { ComputeTarget, Job, JobEvent, PipelineMode } from '@nazar/contracts';
import { API_BASE, request } from './client';

export const jobs = {
  list: (missionId: string, signal?: AbortSignal) =>
    request<Job[]>('GET', `/api/missions/${missionId}/jobs`, undefined, signal),
  get: (id: string, signal?: AbortSignal) =>
    request<Job>('GET', `/api/jobs/${id}`, undefined, signal),
  create: (
    missionId: string,
    pipeline: PipelineMode,
    compute: ComputeTarget = 'local',
    fps?: number,
  ) =>
    request<Job>('POST', `/api/missions/${missionId}/jobs`, {
      pipeline,
      compute,
      ...(fps === undefined ? {} : { fps }),
    }),
  events: (id: string, after: number) =>
    request<JobEvent[]>('GET', `/api/jobs/${id}/events?after=${String(after)}`),
  cancel: (id: string) => request<Job>('POST', `/api/jobs/${id}/cancel`),
  retry: (id: string) => request<Job>('POST', `/api/jobs/${id}/retry`),
  artifactUrl: (artifactId: string) => `${API_BASE}/api/artifacts/${artifactId}/content`,
};
