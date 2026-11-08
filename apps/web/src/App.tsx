import { useEffect } from 'react';
import { useApiStatus } from './api/useApiStatus';
import { useJob } from './api/useJob';
import { useMissions } from './api/useMissions';
import { usePreflight } from './api/usePreflight';
import { useSession } from './api/useSession';
import { AuthCard } from './components/AuthCard';
import { Header } from './components/Header';
import { JobPanel } from './components/JobPanel';
import { MissionPanel } from './components/MissionPanel';
import { PipelinePlan } from './components/PipelinePlan';
import { PreflightPanel } from './components/PreflightPanel';
import { ReadinessPanel } from './components/ReadinessPanel';
import { Card } from './components/ui';
import { Viewport } from './components/Viewport';
import { displayCloud, displayMesh, displaySplat } from './lib/jobState';
import { useViewerStore } from './state/viewerStore';

export function App() {
  const status = useApiStatus();
  const reachable = status.state === 'ready' || status.state === 'not-ready';
  const { session, signedIn, signOut } = useSession(reachable);
  const missions = useMissions(session.state === 'signed-in');
  const readyVideo = missions.selected?.video?.status === 'ready' ? missions.selected.video : null;
  const signedInMission = session.state === 'signed-in' ? (missions.selected?.id ?? null) : null;
  const job = useJob(signedInMission);
  const preflight = usePreflight(readyVideo?.id ?? null);
  const setLayers = useViewerStore((s) => s.setLayers);
  const cloud = displayCloud(job.job);
  const path = job.job?.artifacts.find((a) => a.kind === 'camera_trajectory') ?? null;
  const mesh = displayMesh(job.job);
  const splat = displaySplat(job.job);

  useEffect(() => {
    setLayers(cloud, path, mesh, splat);
  }, [cloud, path, mesh, splat, setLayers]);

  return (
    <>
      <a
        href="#main"
        className="sr-only rounded-button bg-carbon px-4 py-2 text-paper focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-10"
      >
        Skip to content
      </a>
      <Header
        status={status}
        user={session.state === 'signed-in' ? session.user : null}
        onSignOut={() => void signOut()}
      />
      <main
        id="main"
        className="mx-auto flex max-w-page flex-col gap-10 px-4 pb-20 sm:px-6 lg:gap-20"
      >
        <section aria-labelledby="hero-heading" className="flex flex-col gap-6 pt-4 lg:pt-8">
          <span className="mono-label text-slate">Mission control / 00</span>
          <h1 id="hero-heading" className="display max-w-[11ch]">
            One pass. Every angle.
          </h1>
          <p className="max-w-xl text-lg text-slate">
            Upload a single drone video. NAZAR reconstructs the scene in 3D and reveals geometry as
            each pipeline stage produces it. Telemetry is optional.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div className="flex flex-col gap-6">
            {session.state === 'signed-in' ? (
              <MissionPanel
                list={missions.list}
                selected={missions.selected}
                error={missions.error}
                onSelect={missions.select}
                onCreate={missions.create}
                onChange={() => {
                  missions.refresh();
                }}
              />
            ) : session.state === 'setup' || session.state === 'signed-out' ? (
              <AuthCard mode={session.state} onSignedIn={signedIn} />
            ) : (
              <Card id="missions">
                <h2 className="mono-label text-slate">Missions</h2>
                <p className="mt-4 text-slate">
                  Missions appear once the API is reachable. Check System readiness below.
                </p>
              </Card>
            )}
            {signedInMission && (
              <JobPanel
                job={job.job}
                loaded={job.loaded}
                canStart={readyVideo !== null}
                stream={job.stream}
                error={job.error}
                workers={
                  status.state === 'ready' || status.state === 'not-ready'
                    ? (status.readiness.workers ?? [])
                    : []
                }
                report={preflight}
                onStart={job.start}
                onCancel={job.cancel}
                onRetry={job.retry}
              />
            )}
            <ReadinessPanel status={status} />
          </div>
          {/* Stays in view and keeps a fixed height while the (long) results column scrolls. */}
          <div className="lg:sticky lg:top-4 lg:h-[min(calc(100vh-2rem),900px)] lg:self-start">
            <Viewport />
          </div>
        </div>

        {readyVideo && <PreflightPanel key={readyVideo.id} assetId={readyVideo.id} />}

        <PipelinePlan job={job.job} />
      </main>
      <footer className="inverted bg-carbon text-smoke">
        <div className="mx-auto flex max-w-page flex-wrap justify-between gap-4 px-4 py-8 sm:px-6">
          <span className="mono-label">NAZAR · single-pass drone reconstruction</span>
          <span className="mono-label">Outputs are local-frame unless georeferenced</span>
        </div>
      </footer>
    </>
  );
}
