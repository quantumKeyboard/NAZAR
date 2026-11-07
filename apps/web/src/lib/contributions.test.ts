import { describe, expect, it } from 'vitest';
import { describeContributions, gb, type Contribution } from './contributions';

const report: Record<string, Contribution> = {
  dynamic_analysis: {
    applied: true,
    what: 'Masks',
    detections: 0,
    objects: { moving: 0, static: 0 },
    masked_pixel_share: 0,
    peak_vram_mib: 690,
  },
  geometry: { applied: true, placed_keyframes: 22, pixels_kept: 610132, passes: 5 },
  depth_priors: {
    applied: true,
    model: 'vit_large',
    normals_facing_camera: 1,
    scale: { metres_per_unit_estimated: 6.84 },
  },
  sensor_optimization: { applied: false, reason: 'No telemetry was supplied with this mission.' },
  gaussian_splat: { applied: true, gaussians: 276531, iterations: 4000, test_psnr_db: 25.9 },
  surface_processing: {
    applied: true,
    triangles: 1284857,
    vertices: 642369,
    inferred_vertex_share: 0.1359,
    closed_solid: true,
    texture: { size: [2048, 1536], views: 30, coverage: 0.93 },
  },
};

describe('describeContributions', () => {
  const views = describeContributions(report);
  const by = (k: string) => views.find((v) => v.key === k);

  it('keeps the pipeline order and skips what the report lacks', () => {
    expect(views.map((v) => v.key)).toEqual([
      'dynamic_analysis',
      'geometry',
      'depth_priors',
      'sensor_optimization',
      'gaussian_splat',
      'surface_processing',
    ]);
  });

  it('says plainly when nothing moved', () => {
    expect(by('dynamic_analysis')?.headline).toMatch(/No moving objects/);
    expect(by('dynamic_analysis')?.peakVramMib).toBe(690);
  });

  it('quotes real numbers', () => {
    expect(by('geometry')?.headline).toBe('22 camera positions and 610,132 confident points');
    expect(by('surface_processing')?.headline).toBe(
      '1,284,857 triangles, 13.6 % of the surface inferred',
    );
    expect(by('surface_processing')?.facts).toContainEqual(['Closed solid', 'yes']);
    expect(by('surface_processing')?.facts).toContainEqual([
      'Texture',
      '2048 × 1536 px from 30 frames',
    ]);
    expect(by('gaussian_splat')?.facts[0]).toEqual(['Held-out frames PSNR', '25.9 dB']);
  });

  it('shows the implied scale as not applied', () => {
    expect(by('depth_priors')?.facts.at(-1)).toEqual([
      'Implied scale (not applied)',
      '1 unit = 6.84 m',
    ]);
  });

  it('a skipped model carries its reason, not a result', () => {
    const skipped = by('sensor_optimization');
    expect(skipped?.applied).toBe(false);
    expect(skipped?.reason).toBe('No telemetry was supplied with this mission.');
    expect(skipped?.facts).toEqual([]);
  });

  it('handles a missing report', () => {
    expect(describeContributions(undefined)).toEqual([]);
  });

  it('formats memory', () => {
    expect(gb(4769)).toBe('4.7 GB');
  });
});
