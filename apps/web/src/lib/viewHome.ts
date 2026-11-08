import type { Vec3 } from './measure';

/**
 * The model only exists on the side the cameras saw. These helpers pick the starting view
 * from the camera path and the angles the viewer may orbit through, so a first look is never
 * from behind a surface that was not observed. The user can lift the limit ("Free orbit").
 */

/** How far either side of the home direction the limited orbit may swing (radians, 60 deg). */
export const ORBIT_HALF_RANGE = Math.PI / 3;
/** Polar bounds: never straight down the pole, never below the horizon. */
export const POLAR_MIN = 0.15;
export const POLAR_MAX = 1.45;

/** Unit horizontal direction from the scene centre towards the cameras, or null. */
export function homeDirection(cameras: Vec3[], center: Vec3): Vec3 | null {
  if (cameras.length === 0) return null;
  let x = 0;
  let z = 0;
  for (const c of cameras) {
    x += c[0] - center[0];
    z += c[2] - center[2];
  }
  x /= cameras.length;
  z /= cameras.length;
  const length = Math.hypot(x, z);
  // Cameras centred on the scene (an orbit) give no preferred side.
  return length < 1e-6 ? null : [x / length, 0, z / length];
}

/** Azimuth limits around a home direction, or null when they would wrap around +-pi. */
export function azimuthLimits(dir: Vec3): { min: number; max: number } | null {
  const azimuth = Math.atan2(dir[0], dir[2]);
  const min = azimuth - ORBIT_HALF_RANGE;
  const max = azimuth + ORBIT_HALF_RANGE;
  return min < -Math.PI || max > Math.PI ? null : { min, max };
}
