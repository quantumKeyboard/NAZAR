import type { ArtifactManifest } from '@nazar/contracts';
import { Line } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import { useEffect, useState } from 'react';
import {
  Color,
  Float32BufferAttribute,
  Mesh,
  MeshBasicMaterial,
  SRGBColorSpace,
  Vector3,
  type BufferAttribute,
  type BufferGeometry,
  type Material,
} from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { jobs } from '../api/jobs';
import type { Vec3 } from '../lib/measure';
import { confidenceColors, countConfidence } from '../lib/confidence';
import { homeDirection } from '../lib/viewHome';
import { useViewerStore } from '../state/viewerStore';

async function fetchArtifact(id: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(jobs.artifactUrl(id), { credentials: 'same-origin', signal });
  if (!response.ok) throw new Error(`Artifact request failed (${String(response.status)}).`);
  return response.arrayBuffer();
}

interface ParsedCloud {
  geometry: BufferGeometry;
  rgb: BufferAttribute;
  /** Vertex colours by confidence class, when the cloud carries the labels. */
  byConfidence: Float32BufferAttribute | null;
}

/**
 * Renders one point-cloud artifact. A new revision replaces the geometry in place (the old
 * GPU buffer is disposed); requests for superseded revisions are aborted, so a late response
 * can never overwrite a newer cloud.
 */
export function PointCloudLayer({
  manifest,
  visible = true,
}: {
  manifest: ArtifactManifest;
  visible?: boolean;
}) {
  const [cloud, setCloud] = useState<ParsedCloud | null>(null);
  const colorBy = useViewerStore((s) => s.colorBy);
  const measuring = useViewerStore((s) => s.measuring);
  const pick = useViewerStore((s) => s.pick);
  const setLoaded = useViewerStore((s) => s.setLoaded);
  const setLayerError = useViewerStore((s) => s.setLayerError);

  useEffect(() => {
    const controller = new AbortController();
    fetchArtifact(manifest.id, controller.signal)
      .then((buffer) => {
        const loader = new PLYLoader();
        loader.setCustomPropertyNameMapping({ confidence: ['confidence'] });
        const parsed = loader.parse(buffer);
        const classes = parsed.hasAttribute('confidence')
          ? (parsed.getAttribute('confidence').array as ArrayLike<number>)
          : null;
        setCloud({
          geometry: parsed,
          rgb: parsed.getAttribute('color') as BufferAttribute,
          byConfidence: classes ? new Float32BufferAttribute(confidenceColors(classes), 3) : null,
        });
        setLayerError(null);
        setLoaded({
          id: manifest.id,
          kind: manifest.kind,
          revision: manifest.revision,
          points: parsed.getAttribute('position').count,
          confidence: classes ? countConfidence(classes) : null,
        });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setLayerError(err instanceof Error ? err.message : 'Point cloud failed to load.');
      });
    return () => {
      controller.abort();
    };
  }, [manifest.id, manifest.kind, manifest.revision, setLoaded, setLayerError]);

  useEffect(
    () => () => {
      cloud?.geometry.dispose();
    },
    [cloud],
  );

  useEffect(() => {
    if (!cloud) return;
    const wanted = colorBy === 'confidence' && cloud.byConfidence ? cloud.byConfidence : cloud.rgb;
    if (cloud.geometry.getAttribute('color') !== wanted)
      cloud.geometry.setAttribute('color', wanted);
  }, [cloud, colorBy]);

  if (!cloud) return null;
  return (
    <points
      geometry={cloud.geometry}
      visible={visible}
      onClick={(e) => {
        if (!measuring || !visible) return;
        e.stopPropagation();
        // Snap to the clicked point itself, not the ray's closest approach to it.
        const position = cloud.geometry.getAttribute('position');
        const i = e.index ?? -1;
        if (i >= 0) pick([position.getX(i), position.getY(i), position.getZ(i)]);
      }}
    >
      <pointsMaterial size={0.06} sizeAttenuation vertexColors />
    </points>
  );
}

/** sRGB byte -> linear float, so the mesh's vertex colours match the photographed colours. */
const SRGB_TO_LINEAR = (() => {
  const table = new Float32Array(256);
  const c = new Color();
  for (let i = 0; i < 256; i++) {
    c.setRGB(i / 255, i / 255, i / 255, SRGBColorSpace);
    table[i] = c.r;
  }
  return table;
})();

function linearColors(source: BufferAttribute): Float32BufferAttribute {
  const out = new Float32Array(source.count * 3);
  const unit = (v: number) => (source.normalized ? v : v / 255);
  for (let i = 0; i < source.count; i++) {
    out[i * 3] = SRGB_TO_LINEAR[Math.round(unit(source.getX(i)) * 255)] ?? 0;
    out[i * 3 + 1] = SRGB_TO_LINEAR[Math.round(unit(source.getY(i)) * 255)] ?? 0;
    out[i * 3 + 2] = SRGB_TO_LINEAR[Math.round(unit(source.getZ(i)) * 255)] ?? 0;
  }
  return new Float32BufferAttribute(out, 3);
}

/**
 * The solid surface (GLB triangles). Either photographed - a texture baked from the video frames,
 * shown unlit so the picture is not re-shaded - or, when no texture could be made, lit with
 * vertex colours so its shape still reads. Parts the cameras never saw are inferred surface and
 * are described in the report, not hidden here.
 */
export function MeshLayer({
  manifest,
  visible = true,
}: {
  manifest: ArtifactManifest;
  visible?: boolean;
}) {
  const [parts, setParts] = useState<{
    geometry: BufferGeometry;
    material: Material | null;
  } | null>(null);
  const measuring = useViewerStore((s) => s.measuring);
  const pick = useViewerStore((s) => s.pick);
  const setLayerInfo = useViewerStore((s) => s.setLayerInfo);
  const setLayerError = useViewerStore((s) => s.setLayerError);

  useEffect(() => {
    const controller = new AbortController();
    fetchArtifact(manifest.id, controller.signal)
      .then((buffer) => {
        new GLTFLoader().parse(
          buffer,
          '',
          (gltf) => {
            const found: Mesh[] = [];
            gltf.scene.traverse((o) => {
              if (o instanceof Mesh) found.push(o as Mesh);
            });
            const first = found[0];
            if (!first || controller.signal.aborted) return;
            const geo = first.geometry;
            // A photographic texture arrives as the file's own (unlit) material; without one the
            // vertex colours are used with the lit material below.
            const loaded = Array.isArray(first.material) ? first.material[0] : first.material;
            const photo = loaded instanceof MeshBasicMaterial && loaded.map ? loaded : null;
            if (photo?.map) photo.map.anisotropy = 8;
            const color = geo.getAttribute('color') as BufferAttribute | undefined;
            if (color) geo.setAttribute('color', linearColors(color));
            if (!(geo.getAttribute('normal') as BufferAttribute | undefined))
              geo.computeVertexNormals();
            geo.computeBoundingSphere();
            setParts({ geometry: geo, material: photo });
            setLayerInfo({
              triangles: geo.index ? geo.index.count / 3 : geo.getAttribute('position').count / 3,
            });
          },
          () => {
            if (!controller.signal.aborted) setLayerError('The surface mesh could not be read.');
          },
        );
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setLayerError(err instanceof Error ? err.message : 'Mesh failed to load.');
      });
    return () => {
      controller.abort();
    };
  }, [manifest.id, manifest.revision, setLayerInfo, setLayerError]);

  useEffect(
    () => () => {
      parts?.geometry.dispose();
      if (parts?.material instanceof MeshBasicMaterial) parts.material.map?.dispose();
      parts?.material?.dispose();
    },
    [parts],
  );

  if (!parts) return null;
  return (
    <mesh
      geometry={parts.geometry}
      visible={visible}
      onClick={(e) => {
        if (!measuring || !visible) return;
        e.stopPropagation();
        pick([e.point.x, e.point.y, e.point.z]);
      }}
    >
      {parts.material ? (
        <primitive object={parts.material} attach="material" />
      ) : (
        <meshStandardMaterial vertexColors roughness={0.95} metalness={0} side={2} />
      )}
    </mesh>
  );
}

interface TrajectoryDoc {
  poses: { position: [number, number, number] }[];
}

/** Camera positions from a `camera_trajectory` artifact, drawn as a path with markers. */
export function TrajectoryLayer({ manifest }: { manifest: ArtifactManifest }) {
  const [points, setPoints] = useState<[number, number, number][] | null>(null);
  const bounds = useViewerStore((s) => s.cloud?.bounds ?? s.mesh?.bounds ?? null);
  const setHome = useViewerStore((s) => s.setHome);

  useEffect(() => {
    const controller = new AbortController();
    fetchArtifact(manifest.id, controller.signal)
      .then((buffer) => {
        const doc = JSON.parse(new TextDecoder().decode(buffer)) as TrajectoryDoc;
        setPoints(doc.poses.map((p) => p.position));
      })
      .catch(() => undefined);
    return () => {
      controller.abort();
    };
  }, [manifest.id]);

  useEffect(() => {
    if (!points || !bounds) return;
    const centre: Vec3 = [
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2,
    ];
    setHome(homeDirection(points, centre));
  }, [points, bounds, setHome]);

  if (!points || points.length < 2) return null;
  return (
    <group>
      <Line points={points} color="#9effc5" lineWidth={1} />
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.025, 8, 8]} />
          <meshBasicMaterial color="#9effc5" />
        </mesh>
      ))}
    </group>
  );
}

/** Moves the camera to frame the artifact's bounds whenever `fitToken` changes. */
export function FitToBounds({ bounds }: { bounds: ArtifactManifest['bounds'] }) {
  const fitToken = useViewerStore((s) => s.fitToken);
  const home = useViewerStore((s) => s.home);
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as {
    target: Vector3;
    update: () => void;
  } | null;

  useEffect(() => {
    if (!bounds || fitToken === 0) return;
    const min = new Vector3(...bounds.min);
    const max = new Vector3(...bounds.max);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const radius = Math.max(max.distanceTo(min) / 2, 0.5);
    // Start on the side the cameras saw, from a modest elevation; else a generic corner.
    const dir = home
      ? new Vector3(home[0], 0.55, home[2]).normalize()
      : new Vector3(1, 0.8, 1.2).normalize();
    camera.position.copy(center.clone().add(dir.multiplyScalar(radius * 2.2)));
    camera.lookAt(center);
    if (controls) {
      controls.target.copy(center);
      controls.update();
    }
    // Only an explicit fit request moves the camera; new revisions with new bounds do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitToken, home, camera, controls]);

  return null;
}

/** The two picked points of a measurement and the segment between them. */
export function MeasureLayer({ picks }: { picks: Vec3[] }) {
  if (picks.length === 0) return null;
  return (
    <group>
      {picks.length === 2 && (
        <Line points={picks} color="#ffffff" lineWidth={2} depthTest={false} renderOrder={2} />
      )}
      {picks.map((p, i) => (
        <mesh key={i} position={p} renderOrder={3}>
          <sphereGeometry args={[0.12, 12, 12]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>
      ))}
    </group>
  );
}
