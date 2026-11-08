import { describe, expect, it } from 'vitest';
import { ORBIT_HALF_RANGE, azimuthLimits, homeDirection } from './viewHome';

describe('homeDirection', () => {
  it('points from the scene centre to the side the cameras are on', () => {
    const dir = homeDirection(
      [
        [0, 5, 10],
        [2, 5, 12],
      ],
      [0, 0, 0],
    );
    expect(dir?.[1]).toBe(0);
    expect(Math.hypot(dir?.[0] ?? 0, dir?.[2] ?? 0)).toBeCloseTo(1);
    expect(dir?.[2]).toBeGreaterThan(0.9);
  });
  it('has no preferred side for cameras around the centre, or none at all', () => {
    expect(
      homeDirection(
        [
          [5, 0, 0],
          [-5, 0, 0],
        ],
        [0, 0, 0],
      ),
    ).toBeNull();
    expect(homeDirection([], [0, 0, 0])).toBeNull();
  });
});

describe('azimuthLimits', () => {
  it('swings equally either side of the home azimuth', () => {
    const lim = azimuthLimits([0, 0, 1]);
    expect(lim?.min).toBeCloseTo(-ORBIT_HALF_RANGE);
    expect(lim?.max).toBeCloseTo(ORBIT_HALF_RANGE);
  });
  it('gives up when the range would wrap behind the scene', () => {
    expect(azimuthLimits([0, 0, -1])).toBeNull();
  });
});
