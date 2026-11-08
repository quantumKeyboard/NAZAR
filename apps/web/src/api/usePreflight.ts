import type { PreflightReport } from '@nazar/contracts';
import { useEffect, useState } from 'react';
import { assets } from './missions';

/** The upload's preflight report (duration, frame counts) for the ready video, or null. */
export function usePreflight(assetId: string | null): PreflightReport | null {
  const [loaded, setLoaded] = useState<{ id: string; report: PreflightReport } | null>(null);

  useEffect(() => {
    if (!assetId) return;
    const controller = new AbortController();
    assets
      .preflight(assetId, controller.signal)
      .then((report) => {
        setLoaded({ id: assetId, report });
      })
      .catch(() => undefined);
    return () => {
      controller.abort();
    };
  }, [assetId]);

  return loaded && loaded.id === assetId ? loaded.report : null;
}
