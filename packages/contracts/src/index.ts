import constants from '../schema/constants.json' with { type: 'json' };
import type { ComputeTarget, JobStatus, StageName } from './generated';

export type * from './generated';

export const EVENT_SCHEMA_VERSION: number = constants.eventSchemaVersion;

/** Processing stages in execution order. */
export const PIPELINE_STAGES = constants.pipelineStages as readonly StageName[];

export const TERMINAL_STATUSES = constants.terminalStatuses as readonly JobStatus[];

const allowedTransitions = constants.allowedTransitions as Record<JobStatus, readonly JobStatus[]>;

export function canTransition(current: JobStatus, target: JobStatus): boolean {
  return allowedTransitions[current].includes(target);
}

/** Frames per second of video a reconstruction samples: limits and default. */
export const FPS = constants.fps as { min: number; max: number; default: number };

/** Fewest frames a job needs, and the most each compute location allows. */
export const FRAME_LIMITS = constants.frames as {
  auto_min: number;
  auto_max: Record<ComputeTarget, number>;
  min: number;
  max: Record<ComputeTarget, number>;
};
