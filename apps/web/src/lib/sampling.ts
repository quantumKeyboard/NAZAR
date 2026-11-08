import { FPS, FRAME_LIMITS, type ComputeTarget, type TimeModel } from '@nazar/contracts';

/**
 * How many frames a job will use at a chosen rate. Mirrors the server's `count_frames`
 * (packages/media/sampling.py): the video is cut into `round(duration * fps)` equal windows,
 * at most the location's cap, and each window gives its sharpest usable frame. The server's
 * answer (`job.sampling`) is authoritative; this is the preview shown before starting.
 */
export interface FramePlan {
  /** The frame count was chosen from the video length, not from a rate. */
  auto: boolean;
  wanted: number;
  frames: number;
  cap: number;
  clamped: boolean;
  tooFew: boolean;
  /** Frames actually used per second of video. */
  effectiveFps: number;
}

/**
 * Frames automatic mode asks for: about one per second of video, never fewer than the
 * minimum (a short clip is sampled densely, not reduced to a few views) and never more than
 * the location's automatic maximum. Mirrors `auto_frames` in packages/media/sampling.py.
 */
export function autoFrames(durationS: number, compute: ComputeTarget): number {
  return Math.min(
    FRAME_LIMITS.auto_max[compute],
    Math.max(FRAME_LIMITS.auto_min, Math.round(durationS)),
  );
}

/** `fps` null = automatic. */
export function planFrames(
  durationS: number,
  fps: number | null,
  compute: ComputeTarget,
): FramePlan {
  const cap = FRAME_LIMITS.max[compute];
  const wanted =
    fps === null ? autoFrames(durationS, compute) : Math.max(1, Math.round(durationS * fps));
  const frames = Math.min(wanted, cap);
  return {
    auto: fps === null,
    wanted,
    frames,
    cap,
    clamped: wanted > cap,
    tooFew: frames < FRAME_LIMITS.min,
    effectiveFps: durationS > 0 ? frames / durationS : 0,
  };
}

export const FPS_PRESETS: readonly number[] = [0.5, 1, 2, 4];

/** A rate typed by the user, kept inside the allowed range; null when it is not a number. */
export function parseFps(text: string): number | null {
  const value = Number(text.trim().replace(',', '.'));
  if (!Number.isFinite(value) || text.trim() === '') return null;
  return clampFps(value);
}

export function clampFps(value: number): number {
  const stepped = Math.round(value * 100) / 100;
  return Math.min(FPS.max, Math.max(FPS.min, stepped));
}

/** The worker's advertised time per frame, applied to a frame count (seconds). */
export function preRunSeconds(model: TimeModel | null | undefined, frames: number): number | null {
  if (!model) return null;
  return model.fixed_s + model.per_frame_s * frames;
}

export function formatFps(fps: number): string {
  return `${String(Math.round(fps * 100) / 100)} fps`;
}
