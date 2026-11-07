import { FPS, FRAME_LIMITS } from '@nazar/contracts';
import { describe, expect, it } from 'vitest';
import { autoFrames, clampFps, formatFps, parseFps, planFrames, preRunSeconds } from './sampling';

describe('planFrames', () => {
  it('one frame per second of video by default', () => {
    expect(FPS.default).toBe(1);
    const p = planFrames(38.4, 1, 'local');
    expect(p.frames).toBe(38);
    expect(p.clamped).toBe(false);
    expect(p.effectiveFps).toBeCloseTo(0.99, 1);
  });

  it('lowers the count to the location cap and says so', () => {
    const p = planFrames(100, 4, 'local');
    expect(p.wanted).toBe(400);
    expect(p.frames).toBe(FRAME_LIMITS.max.local);
    expect(p.clamped).toBe(true);
    expect(planFrames(100, 4, 'college').frames).toBeGreaterThan(p.frames);
  });

  it('flags a rate that gives too few frames', () => {
    expect(planFrames(5, 0.25, 'local').tooFew).toBe(true);
    expect(planFrames(5, 1, 'local').tooFew).toBe(false);
  });
});

describe('fps input', () => {
  it('parses, clamps and rejects', () => {
    expect(parseFps('2')).toBe(2);
    expect(parseFps(' 1,5 ')).toBe(1.5);
    expect(parseFps('0.01')).toBe(FPS.min);
    expect(parseFps('99')).toBe(FPS.max);
    expect(parseFps('')).toBeNull();
    expect(parseFps('abc')).toBeNull();
  });
  it('keeps two decimals', () => {
    expect(clampFps(1.23456)).toBe(1.23);
    expect(formatFps(1)).toBe('1 fps');
    expect(formatFps(0.25)).toBe('0.25 fps');
  });
});

describe('preRunSeconds', () => {
  it('applies the worker time per frame', () => {
    expect(preRunSeconds({ fixed_s: 300, per_frame_s: 3, basis: 'x' }, 40)).toBe(420);
    expect(preRunSeconds(null, 40)).toBeNull();
  });
});

describe('automatic frame count', () => {
  it('never starves a short clip', () => {
    expect(autoFrames(6.6, 'local')).toBe(FRAME_LIMITS.auto_min);
    const p = planFrames(6.6, null, 'local');
    expect(p.auto).toBe(true);
    expect(p.frames).toBe(FRAME_LIMITS.auto_min);
    expect(p.effectiveFps).toBeGreaterThan(3);
  });
  it('follows the length of a longer clip up to the location maximum', () => {
    expect(autoFrames(30, 'local')).toBe(30);
    expect(autoFrames(500, 'local')).toBe(FRAME_LIMITS.auto_max.local);
    expect(autoFrames(500, 'college')).toBe(FRAME_LIMITS.auto_max.college);
    expect(planFrames(500, null, 'local').clamped).toBe(false);
  });
  it('a manual rate is not automatic', () => {
    expect(planFrames(6.6, 1, 'local').auto).toBe(false);
    expect(planFrames(6.6, 1, 'local').frames).toBe(7);
  });
});
