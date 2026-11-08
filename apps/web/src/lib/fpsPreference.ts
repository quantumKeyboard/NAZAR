import { FPS } from '@nazar/contracts';
import { clampFps } from './sampling';

/**
 * The frame rate last chosen in this browser: a number of frames per second, or null for
 * automatic (the default: the frame count follows the video's length).
 */
// v2: earlier versions stored a plain rate here (default 1 fps); automatic is now the default,
// so an old value must not silently turn automatic mode back into 1 fps.
const KEY = 'nazar.fps.v2';

export function loadFps(): number | null {
  try {
    const stored = window.localStorage.getItem(KEY);
    if (stored === null || stored === 'auto') return null;
    const value = Number(stored);
    return value > 0 ? clampFps(value) : null;
  } catch {
    return null;
  }
}

export function saveFps(fps: number | null) {
  try {
    window.localStorage.setItem(KEY, fps === null ? 'auto' : String(fps));
  } catch {
    // A convenience only; the choice still applies to this page view.
  }
}

/** The rate shown when the user switches from automatic to a manual rate. */
export const MANUAL_DEFAULT = FPS.default;
