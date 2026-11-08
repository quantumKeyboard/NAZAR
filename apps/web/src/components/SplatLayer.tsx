import type { ArtifactManifest } from '@nazar/contracts';
import { useThree } from '@react-three/fiber';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { useEffect, useState } from 'react';
import { jobs } from '../api/jobs';
import { useViewerStore } from '../state/viewerStore';

/**
 * The photoreal Gaussian splat (PLY). Spark sorts and draws the Gaussians; it shares the
 * scene's camera and frame with the points and the mesh, so they line up exactly.
 */
export default function SplatLayer({
  manifest,
  visible = true,
}: {
  manifest: ArtifactManifest;
  visible?: boolean;
}) {
  const gl = useThree((s) => s.gl);
  const setLayerInfo = useViewerStore((s) => s.setLayerInfo);
  const setLayerError = useViewerStore((s) => s.setLayerError);
  // Created in an effect, not memoised: a disposed renderer must never be reused (React's
  // development double-invoke runs the cleanup once before the layer is really mounted).
  const [spark, setSpark] = useState<SparkRenderer | null>(null);
  const [splat, setSplat] = useState<SplatMesh | null>(null);

  useEffect(() => {
    const created = new SparkRenderer({ renderer: gl });
    // An external GPU object with a lifetime: it is created and disposed with this effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSpark(created);
    return () => {
      setSpark(null);
      created.dispose();
    };
  }, [gl]);

  useEffect(() => {
    const controller = new AbortController();
    let created: SplatMesh | null = null;
    fetch(jobs.artifactUrl(manifest.id), { credentials: 'same-origin', signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`Splat request failed (${String(r.status)}).`);
        return r.arrayBuffer();
      })
      .then(async (bytes) => {
        const mesh = new SplatMesh({ fileBytes: bytes, fileName: 'scene.ply' });
        created = mesh;
        await mesh.initialized;
        if (controller.signal.aborted) {
          mesh.dispose();
          return;
        }
        setSplat(mesh);
        setLayerInfo({ splats: mesh.packedSplats?.numSplats ?? null });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setLayerError(err instanceof Error ? err.message : 'The splat could not be loaded.');
      });
    return () => {
      controller.abort();
      created?.dispose();
      setSplat(null);
    };
  }, [manifest.id, manifest.revision, setLayerInfo, setLayerError]);

  return (
    <>
      {spark && <primitive object={spark} />}
      {spark && splat && <primitive object={splat} visible={visible} />}
    </>
  );
}
