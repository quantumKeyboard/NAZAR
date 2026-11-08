import { lazy, Suspense } from 'react';
import { CONFIDENCE_CLASSES } from '../lib/confidence';
import { distance, distanceText } from '../lib/measure';
import { sourceText } from '../lib/provenance';
import { useViewerStore } from '../state/viewerStore';
import { Button, Card, Tag } from './ui';

const ViewerScene = lazy(() => import('./ViewerScene'));

const FRAME_TEXT = {
  local_reconstruction: 'Local',
  enu: 'ENU',
  ecef: 'ECEF',
  geographic: 'Geographic',
  projected: 'Projected',
} as const;

const SCALE_TEXT = {
  unknown: 'unknown scale',
  relative: 'relative scale',
  metric_estimated: 'metric (estimated)',
  metric_sensor: 'metric (sensor)',
} as const;

export function Viewport() {
  const s = useViewerStore();
  const { cloud, path, loaded, mesh, splat, layerInfo } = s;
  const kindText = loaded?.kind === 'point_cloud' ? 'Final cloud' : 'Preview';
  const confidence = loaded?.confidence ?? null;
  const byConfidence = s.colorBy === 'confidence' && confidence !== null;
  const [a, b] = s.picks;
  const measureText = !s.measuring
    ? null
    : a && b
      ? `Distance: ${distanceText(distance(a, b), cloud?.frame)}`
      : a
        ? 'Click a second point.'
        : 'Click a point in the cloud to start measuring.';

  return (
    <Card
      inverted
      id="viewer"
      aria-labelledby="viewer-heading"
      className="flex h-full min-h-[420px] flex-col gap-4 lg:min-h-[560px]"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="viewer-heading" className="mono-label text-smoke">
          3D viewer
        </h2>
        <Tag tone="mint">{sourceText(cloud)}</Tag>
      </div>

      <div
        className="relative min-h-[380px] flex-1 overflow-hidden rounded-card-sm bg-carbon"
        data-testid="viewer-stage"
        data-cloud-kind={loaded?.kind ?? ''}
        data-cloud-revision={loaded?.revision ?? 0}
        data-cloud-points={loaded?.points ?? 0}
      >
        <Suspense fallback={<p className="mono-label p-6 text-smoke">Loading 3D engine…</p>}>
          <ViewerScene />
        </Suspense>
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-1 p-4">
          {loaded ? (
            <p
              className="font-mono text-xs text-smoke"
              aria-live="polite"
              data-testid="viewer-loaded"
            >
              {kindText} · revision {loaded.revision} · {loaded.points.toLocaleString()} points
              {layerInfo.triangles !== null &&
                ` · mesh ${Math.round(layerInfo.triangles).toLocaleString()} triangles`}
              {layerInfo.splats !== null &&
                ` · splat ${layerInfo.splats.toLocaleString()} Gaussians`}
            </p>
          ) : null}
          {s.limitOrbit && s.home && (mesh ?? cloud) ? (
            <p className="font-mono text-xs text-smoke" data-testid="orbit-note">
              Viewing the side the camera saw. The far side was never filmed, so it is not in the
              model. Free orbit shows it anyway.
            </p>
          ) : null}
          {splat && s.showSplat && layerInfo.splats === null && !s.layerError ? (
            <p className="font-mono text-xs text-smoke" aria-live="polite">
              Loading the photoreal splat…
            </p>
          ) : null}
          {measureText ? (
            <p
              className="font-mono text-xs text-paper"
              aria-live="polite"
              data-testid="measure-readout"
            >
              {measureText}
            </p>
          ) : null}
          {byConfidence ? (
            <ul
              className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-paper"
              aria-label="Point colours: confidence class"
              data-testid="confidence-legend"
            >
              {CONFIDENCE_CLASSES.filter((c) => confidence[c.key] > 0).map((c) => (
                <li key={c.key} className="flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="inline-block size-2.5 rounded-full"
                    style={{ backgroundColor: c.hex }}
                  />
                  {c.label} {confidence[c.key].toLocaleString()}
                </li>
              ))}
              <li className="text-smoke">heuristic labels; rule in the job report</li>
            </ul>
          ) : null}
          {loaded ? null : (
            <>
              <p className="text-lg font-medium text-paper" data-testid="viewer-empty">
                No geometry yet.
              </p>
              <p className="max-w-md text-sm text-smoke">
                Point clouds appear here as soon as a job publishes its first one. The grid and axes
                are reference aids, not data.
              </p>
            </>
          )}
          {s.layerError && (
            <p role="alert" className="text-sm text-paper">
              {s.layerError}
            </p>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <dl className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs">
          <div className="flex gap-2">
            <dt className="text-smoke uppercase">Frame</dt>
            <dd>{cloud?.frame ? FRAME_TEXT[cloud.frame.type] : 'Local · none'}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="text-smoke uppercase">Units</dt>
            <dd>
              {cloud?.frame
                ? cloud.frame.metres_per_unit
                  ? `1 unit = ${cloud.frame.metres_per_unit.toFixed(2)} m, ${SCALE_TEXT[cloud.frame.scale_status]}`
                  : `${cloud.frame.units}, ${SCALE_TEXT[cloud.frame.scale_status]}`
                : 'Unknown'}
            </dd>
          </div>
        </dl>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Viewer controls">
          {splat && (
            <Button
              variant="ghost-inverted"
              aria-pressed={s.showSplat}
              onClick={s.toggleSplat}
              data-testid="toggle-splat"
            >
              Photoreal splat (front only)
            </Button>
          )}
          {mesh && (
            <Button
              variant="ghost-inverted"
              aria-pressed={s.showMesh}
              onClick={s.toggleMesh}
              data-testid="toggle-mesh"
            >
              Solid mesh
            </Button>
          )}
          {cloud && (
            <Button variant="ghost-inverted" aria-pressed={s.showCloud} onClick={s.toggleCloud}>
              Points
            </Button>
          )}
          {confidence && (
            <Button
              variant="ghost-inverted"
              aria-pressed={s.colorBy === 'confidence'}
              onClick={() => {
                s.setColorBy(s.colorBy === 'confidence' ? 'rgb' : 'confidence');
              }}
            >
              Colour by confidence
            </Button>
          )}
          {(cloud ?? mesh) && (
            <Button variant="ghost-inverted" aria-pressed={s.measuring} onClick={s.toggleMeasure}>
              Measure
            </Button>
          )}
          {path && (
            <Button variant="ghost-inverted" aria-pressed={s.showPath} onClick={s.togglePath}>
              Camera path
            </Button>
          )}
          {(cloud ?? mesh) && (
            <Button
              variant="ghost-inverted"
              aria-pressed={!s.limitOrbit}
              onClick={s.toggleLimitOrbit}
              data-testid="toggle-free-orbit"
            >
              Free orbit
            </Button>
          )}
          <Button variant="ghost-inverted" aria-pressed={s.showGrid} onClick={s.toggleGrid}>
            Grid
          </Button>
          <Button variant="ghost-inverted" aria-pressed={s.showAxes} onClick={s.toggleAxes}>
            Axes
          </Button>
          {(cloud ?? mesh ?? splat) && (
            <Button variant="ghost-inverted" onClick={s.requestFit}>
              Fit
            </Button>
          )}
          <Button variant="ghost-inverted" onClick={s.requestReset}>
            Reset view
          </Button>
        </div>
      </div>
    </Card>
  );
}
