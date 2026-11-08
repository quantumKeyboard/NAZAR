import { Grid, OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { lazy, Suspense, useEffect, useRef, type ComponentRef } from 'react';
import { azimuthLimits, POLAR_MAX, POLAR_MIN } from '../lib/viewHome';
import { useViewerStore } from '../state/viewerStore';
import { FitToBounds, MeasureLayer, MeshLayer, PointCloudLayer, TrajectoryLayer } from './layers';

// Spark is large; it is only fetched once a job actually has a splat.
const SplatLayer = lazy(() => import('./SplatLayer'));

const HOME_POSITION = [6, 4.5, 8] as const;

/**
 * R3F scene. Reference aids (grid, axes) are always local-frame helpers, not data.
 * Reconstruction layers mount only when a real artifact manifest exists.
 */
export default function ViewerScene() {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const showGrid = useViewerStore((s) => s.showGrid);
  const showAxes = useViewerStore((s) => s.showAxes);
  const showCloud = useViewerStore((s) => s.showCloud);
  const showPath = useViewerStore((s) => s.showPath);
  const showMesh = useViewerStore((s) => s.showMesh);
  const showSplat = useViewerStore((s) => s.showSplat);
  const mesh = useViewerStore((s) => s.mesh);
  const splat = useViewerStore((s) => s.splat);
  const resetToken = useViewerStore((s) => s.resetToken);
  const limitOrbit = useViewerStore((s) => s.limitOrbit);
  const home = useViewerStore((s) => s.home);
  const cloud = useViewerStore((s) => s.cloud);
  const path = useViewerStore((s) => s.path);
  const picks = useViewerStore((s) => s.picks);

  const limits = limitOrbit && home && (mesh ?? cloud) ? azimuthLimits(home) : null;

  useEffect(() => {
    if (resetToken > 0) controls.current?.reset();
  }, [resetToken]);

  return (
    <Canvas
      camera={{ position: [...HOME_POSITION], fov: 45, near: 0.05, far: 500 }}
      dpr={[1, 2]}
      onCreated={({ raycaster }) => {
        // Scene units are normalised (median camera distance = 10), so a fixed pick radius works.
        raycaster.params.Points.threshold = 0.08;
      }}
      gl={{ antialias: true }}
      aria-label={
        cloud
          ? '3D viewport showing the current point cloud'
          : '3D viewport: empty local frame with reference grid and axes'
      }
      data-testid="viewer-canvas"
    >
      <color attach="background" args={['#000000']} />
      {showGrid && (
        <Grid
          args={[20, 20]}
          cellSize={0.5}
          cellThickness={0.6}
          cellColor="#2f2f2f"
          sectionSize={2.5}
          sectionThickness={1}
          sectionColor="#444444"
          fadeDistance={30}
          infiniteGrid
        />
      )}
      {showAxes && <axesHelper args={[1.5]} />}
      <hemisphereLight args={['#ffffff', '#444444', 1.6]} />
      <directionalLight position={[5, 8, 6]} intensity={1.8} />
      {cloud && <PointCloudLayer key={cloud.job_id} manifest={cloud} visible={showCloud} />}
      {mesh && <MeshLayer key={`mesh-${mesh.id}`} manifest={mesh} visible={showMesh} />}
      {splat && (
        <Suspense fallback={null}>
          <SplatLayer key={`splat-${splat.id}`} manifest={splat} visible={showSplat} />
        </Suspense>
      )}
      {path && showPath && <TrajectoryLayer manifest={path} />}
      <MeasureLayer picks={picks} />
      {(cloud ?? mesh ?? splat) && (
        <FitToBounds bounds={(cloud ?? mesh ?? splat)?.bounds ?? null} />
      )}
      <OrbitControls
        ref={controls}
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minAzimuthAngle={limits?.min ?? -Infinity}
        maxAzimuthAngle={limits?.max ?? Infinity}
        minPolarAngle={limits ? POLAR_MIN : 0}
        maxPolarAngle={limits ? POLAR_MAX : Math.PI}
      />
    </Canvas>
  );
}
