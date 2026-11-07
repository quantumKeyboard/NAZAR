/**
 * What each model added to the result, read from the processing report
 * (`report.contributions`, written by the worker). Plain numbers and sentences only: a model
 * that did not run says why, never a made-up result.
 */
export type Contribution = Record<string, unknown>;

export interface ContributionView {
  key: string;
  title: string;
  model: string;
  applied: boolean;
  /** One line: what it did for this scene. */
  headline: string;
  /** Why it did not run (when `applied` is false). */
  reason: string | null;
  /** The model's own explanation of its role. */
  what: string | null;
  peakVramMib: number | null;
  facts: [string, string][];
}

const ORDER: { key: string; title: string; model: string }[] = [
  { key: 'dynamic_analysis', title: 'Moving objects', model: 'YOLO11 + SAM 2' },
  { key: 'geometry', title: 'Camera poses and points', model: 'Fast3R' },
  { key: 'depth_priors', title: 'Depth and surface direction', model: 'Metric3D-v2' },
  { key: 'sensor_optimization', title: 'GPS alignment', model: 'Sim(3) fit to telemetry' },
  { key: 'dense_reconstruction', title: 'Fused point cloud', model: 'Fusion of the above' },
  { key: 'gaussian_splat', title: 'Photoreal splat', model: 'gsplat' },
  { key: 'surface_processing', title: 'Solid textured surface', model: 'Height map + texture' },
];

const n = (c: Contribution, key: string): number | null => {
  const v = c[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
};
const int = (v: number | null) => (v === null ? '?' : Math.round(v).toLocaleString('en-US'));
const pct = (v: number | null, digits = 0) => (v === null ? '?' : `${(v * 100).toFixed(digits)} %`);
const str = (c: Contribution, key: string): string | null =>
  typeof c[key] === 'string' ? c[key] : null;

/** The photographic texture, when the surface has one. */
function textureFacts(c: Contribution): [string, string][] {
  const tex = c.texture;
  if (typeof tex !== 'object' || tex === null) return [];
  const t = tex as Record<string, unknown>;
  if (t.applied === false) {
    return [['Texture', typeof t.reason === 'string' ? t.reason : 'not made']];
  }
  const size = Array.isArray(t.size) ? (t.size as unknown[]).join(' × ') : '?';
  const coverage = typeof t.coverage === 'number' ? pct(t.coverage, 0) : '?';
  return [
    ['Texture', `${size} px from ${int(typeof t.views === 'number' ? t.views : null)} frames`],
    ['Surface covered by frames', coverage],
  ];
}

function facts(key: string, c: Contribution): { headline: string; facts: [string, string][] } {
  switch (key) {
    case 'dynamic_analysis': {
      const objects = c.objects as Record<string, unknown> | undefined;
      const moving = typeof objects?.moving === 'number' ? objects.moving : 0;
      const stat = typeof objects?.static === 'number' ? objects.static : 0;
      return {
        headline:
          moving > 0
            ? `${String(moving)} moving object${moving === 1 ? '' : 's'} masked out of every later step`
            : 'No moving objects found, so nothing was masked',
        facts: [
          ['Detections', int(n(c, 'detections'))],
          ['Moving / parked', `${String(moving)} / ${String(stat)}`],
          ['Pixels masked', pct(n(c, 'masked_pixel_share'), 2)],
        ],
      };
    }
    case 'geometry':
      return {
        headline: `${int(n(c, 'placed_keyframes'))} camera positions and ${int(n(c, 'pixels_kept'))} confident points`,
        facts: [
          ['Frames placed', int(n(c, 'placed_keyframes'))],
          ['Points kept', int(n(c, 'pixels_kept'))],
          ['Passes', int(n(c, 'passes'))],
        ],
      };
    case 'depth_priors': {
      const scale = c.scale as Record<string, unknown> | undefined;
      const mpu = typeof scale?.metres_per_unit_estimated === 'number' ? scale : null;
      return {
        headline: `Depth and normals for every frame (${str(c, 'model') ?? 'Metric3D'})`,
        facts: [
          ['Model', str(c, 'model') ?? '?'],
          ['Normals facing camera', pct(n(c, 'normals_facing_camera'))],
          [
            'Implied scale (not applied)',
            mpu ? `1 unit = ${Number(mpu.metres_per_unit_estimated).toFixed(2)} m` : 'n/a',
          ],
        ],
      };
    }
    case 'sensor_optimization':
      return { headline: 'Scaled to metres from the GPS track', facts: [] };
    case 'dense_reconstruction':
      return {
        headline: `${int(n(c, 'final_points'))} points, ${int(n(c, 'from_depth_prior'))} filled from depth`,
        facts: [
          ['Fast3R points kept', int(n(c, 'fast3r_kept'))],
          ['Holes filled by Metric3D', int(n(c, 'depth_prior_filled'))],
          [
            'Depth agrees with Fast3R',
            `${int(n(c, 'prior_views_agreeing'))} of ${int(n(c, 'prior_views_compared'))} frames`,
          ],
          ['Outliers removed', int(n(c, 'outliers_removed'))],
        ],
      };
    case 'gaussian_splat':
      return {
        headline: `${int(n(c, 'gaussians'))} Gaussians trained for ${int(n(c, 'iterations'))} steps`,
        facts: [
          [
            'Held-out frames PSNR',
            n(c, 'test_psnr_db') === null ? '?' : `${String(n(c, 'test_psnr_db'))} dB`,
          ],
          [
            'Training PSNR',
            n(c, 'train_psnr_db') === null ? '?' : `${String(n(c, 'train_psnr_db'))} dB`,
          ],
          [
            'Camera refinement',
            n(c, 'pose_refinement_mean_shift') === null
              ? '?'
              : `${String(n(c, 'pose_refinement_mean_shift'))} units`,
          ],
        ],
      };
    case 'surface_processing':
      return {
        headline: `${int(n(c, 'triangles'))} triangles, ${pct(n(c, 'inferred_vertex_share'), 1)} of the surface inferred`,
        facts: [
          ['Vertices', int(n(c, 'vertices'))],
          ['Triangles', int(n(c, 'triangles'))],
          ['Inferred (not seen)', pct(n(c, 'inferred_vertex_share'), 1)],
          ['Closed solid', c.closed_solid === true || c.watertight === true ? 'yes' : 'no'],
          ...textureFacts(c),
        ],
      };
    default:
      return { headline: 'Applied', facts: [] };
  }
}

export function describeContributions(
  contributions: Record<string, Contribution> | undefined,
): ContributionView[] {
  if (!contributions) return [];
  return ORDER.flatMap(({ key, title, model }) => {
    const c = contributions[key];
    if (!c) return [];
    const applied = c.applied === true;
    const body = applied ? facts(key, c) : { headline: 'Did not run', facts: [] };
    return [
      {
        key,
        title,
        model,
        applied,
        headline: body.headline,
        reason: applied ? null : str(c, 'reason'),
        what: str(c, 'what'),
        peakVramMib: n(c, 'peak_vram_mib'),
        facts: body.facts,
      },
    ];
  });
}

/** GPU memory as the UI quotes it: "4.3 GB". */
export function gb(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GB`;
}
