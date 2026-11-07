# Implementation status

_Last updated: 2026-11-07_

## Latest: **A solid, textured model from a single pass (branch `feature/frontend-pipeline`, 2026-11-07) — written and unit-tested, NOT yet run end to end**

Triggered by a real test: a 6.6 s portrait 1080p clip (Taj Mahal) gave only 7 frames and a lumpy Poisson mesh with vertex colours. Three changes, implemented without running the pipeline on that video:

1. **Automatic frame count (the new default).** About one frame per second but never fewer than 24 (`AUTO_MIN_FRAMES`) and at most 40 local / 96 college (`AUTO_MAX_FRAMES`); a 6.6 s clip now gives 24 frames, not 7. The fps control keeps a manual mode; `POST /jobs` with no `fps` means automatic (`SamplingInfo.auto`, migration `0006`, `--fps` optional in `local_run`). Training images for the splat are shrunk to at most 1920 px on the long side (`--max-side`) so 4K footage cannot exhaust GPU memory.
2. **Surface from a height map, not Poisson.** `apps/worker/src/nazar_worker/heightfield.py`: the fused points are binned on the ground plane, each cell keeps a high percentile of its heights, isolated spikes are removed, small holes are bridged by normalised convolution and labelled _inferred_, and the heights are smoothed with an edge-preserving filter. The grid becomes triangles and is closed into a solid block (side walls and a flat base). It runs on the CPU in the worker; Open3D Poisson remains only as a fallback when the scene cannot be a height map (too sparse, or under 25 % of the surface has points under it).
3. **Real texture.** A top-down texture is baked from the _full-resolution_ frames: each texel is a 3D point on the surface, projected into every frame, checked for occlusion against a depth buffer of the surface, and blended (weights favour near-perpendicular, close, away-from-border views). Texel size follows the frames' own ground resolution (capped at 3072 px). The GLB carries `TEXCOORD_0`, a JPEG texture and `KHR_materials_unlit`; the viewer shows it unlit so the photograph is not re-shaded. If no texture can be made the mesh falls back to vertex colours and the report says why.

Verified by unit tests only (synthetic hill with known colours: heights recovered, bridged hole labelled inferred, floater removed, mesh closed with unit normals and valid UVs, baked texture matches the ground-truth colour function, occlusion and no-camera cases, GLB round trip; media/API/web tests for automatic frames). **Not run:** the pipeline, the browser, the GPU runners, real footage. Timing of the new surface stage (forecast 20 s + 3 s/frame) is an estimate until a real run calibrates it. Known limits: unseen sides cannot be measured (the model is a relief with a flat base); facades become steep slopes and stretch in the texture; exposure differences between frames can leave faint seams; the texture uses the fused cloud's up-axis, so a scene with no ground plane may be mis-oriented.

## Previous: **Frontend connected to the full pipeline (branch `feature/frontend-pipeline`, 2026-11-05)** **Frontend connected to the full pipeline (branch `feature/frontend-pipeline`, 2026-11-05)**

Built: per-job frames-per-second (0.25 to 4, default 1; API samples the sharpest usable frame per time window, caps 64 local / 128 college, 422 below 3 frames); `job.estimate` events and a worker time forecast (cost model fitted to the measured run, calibrated from stored real durations, plus a pre-run `TimeModel` and model readiness advertised through the lease); live run screen (step N of 7, model, counts, elapsed, time-left range, basis); pipeline cards with forecast vs. real time and GPU peak vs. budget; per-model contributions from the report; viewer layers for the Gaussian splat (Spark 2.4.0, lazy chunk) and the vertex-coloured mesh with toggles, mesh measuring, and a sticky fixed-height viewer. Web libs (`sampling`, `estimate`, `contributions`) are unit-tested.

Verified in a real browser run (isolated storage, local GPU worker, `sample1.mp4`, 1 fps, 23 frames): finished in 8.4 min, peak GPU 4.6 GB of 5.5 GB, all six models contributed, no console errors, splat and mesh load and align with the points (splat verified on the RTX 4050 through Chromium/ANGLE).

Found and fixed by that run: the splat never drew (a disposed `SparkRenderer` was reused after React's development double-invoke); the fuse-and-splat bar jumped from 98 % to 6 % and the time left from 26 s to 9 minutes (fusion and splat progress count different things; now mapped to the stage's share); the pre-run estimate was low (4-5 min against 8.4) because the calibration history held splat-less 0.06 s "dense" runs (now kept apart and implausible ratios ignored). The corrected estimate has not been re-timed on a second full run.

Not done: local frame cap of 64 untested at high fps on 6 GB; college profile weights (about 6.5 GB) not downloaded and its forecast unmeasured; Metric3D scale still reported, not applied; no texture baking, so the mesh stays vertex-coloured. See [`DEMO.md`](DEMO.md).

### Viewer polish after first review (2026-11-05)

Feedback: the model looked like points and bumpy mesh from any angle but the filming one, and the grid hurt it. Changes: grid and axes off by default; the solid mesh is shown first (the splat smears into streaks off-axis, so it is labelled "front only"); the orbit is limited to 60 degrees either side of the side the cameras saw (home direction from the camera path), with a **Free orbit** button and a note that the far side was never filmed; the mesh gets Taubin smoothing (20 iterations, volume-preserving) in `surface_runner.py`; camera markers shrunk. Checked by re-running sample1 through the browser and viewing from several angles; e2e 27 passed. Limit that remains: one pass of video gives a one-sided relief, not a closed object. Seen from low angles it is a thin sheet. A closed, all-round model needs footage from more sides, or a generative completion step, which this project does not claim to do.

## Current milestone: **Full model pipeline (branch `feature/full-pipeline`) — video in, explainable 3D model out**

Every model from the brief is now wired in sequence: YOLO11 + SAM 2 (moving-object masks) → Fast3R (poses + dense points) → Metric3D-v2 (depth + normals) → fusion → gsplat Gaussian splat (with pose refinement) → Open3D Poisson mesh. Each model's contribution, peak VRAM and any skip reason is in `report.json` (`contributions`). Two profiles: `local` (6 GB laptop, hard 5.5 GiB budget, a model may be skipped only on out-of-memory) and `clg` (college GPU, strict, larger models). Test it with:

```
uv run --package nazar-worker python scripts/test_full_pipeline.py --video docs/sample/video/sample1.mp4 --how local
```

See "Full pipeline report" below and [`pipeline-completion/06-FULL-PIPELINE-PLAN.md`](pipeline-completion/06-FULL-PIPELINE-PLAN.md).

### Earlier: Milestone 3 (GPU) — Fast3R dense geometry, with the CPU baseline as the visible fallback

**Reconstruct** now runs **Fast3R** on the GPU when the worker has the Fast3R runtime set up (`pnpm gpu:setup`; this laptop's RTX 4050 is set up). Otherwise it runs the classical CPU structure-from-motion baseline. Either way the result is in a local frame with unknown scale, and the job, every artifact and the viewer say which method ran. If Fast3R fails on a job, the job still completes with the CPU baseline and a visible `fast3r_fallback` warning that gives the reason.

_(Superseded by the full pipeline above.)_ The college GPU (Milestone 1) has not been accessed. The synthetic test pipeline still exists for plumbing checks and stays labelled as synthetic.

**Optional telemetry** (DJI SRT / CSV) can now be attached to a mission. When the camera path matches the GPS track well enough, the result gets a metric scale from GNSS; otherwise it stays local and says why. See the telemetry report below.

Checks that need you (college GPU, sound and vibration, phone, decisions) are listed in [`PENDING_USER_TESTS.md`](PENDING_USER_TESTS.md).

The user had no third-party API keys yet. None are needed so far:

- Machine secrets (`NAZAR_SECRET_KEY`, `NAZAR_WORKER_TOKEN`) are generated locally by `pnpm env:init` into the git-ignored `.env`. The script prints only the variable names.
- The operator account is created in the web app on first run. `NAZAR_BOOTSTRAP_ADMIN_*` became optional, for headless use.
- All API-key placeholders in `.env.example` stay blank.

Milestone 1 (college VDI hardware validation) still needs the user inside a SyncVDI session. It has not started.

## Done

| Area | What exists |
|---|---|
| Monorepo | pnpm 12 + Turborepo 2. uv workspace (Python 3.12). ADR-0001. |
| Contracts | Job state machine, events, artifact manifest, health/readiness (Stage 1). **New:** auth state, credentials, missions, assets (status + progress), preflight report (probe, analysis info, summary, keyframes, warnings, columnar per-frame series). Pydantic → JSON Schema → TS with drift check. |
| Media (`packages/media`, new) | ffprobe parsing (rotation, display size, attached pictures ignored). Packet timestamps; VFR decided from real intervals. Single-pass FFmpeg decode at 320 px gray via argv (no shell). Per-frame sharpness (Laplacian variance), brightness and clipping, change, sub-pixel phase-correlation motion, 48-px viewpoint signature. Flags: blurry, dark, over-exposed, repeated. Keyframe selection by viewpoint novelty with min/max gap, sharpest-in-look-ahead, and a cap of 96. Full-res + thumbnail JPEG extraction, atomic writes, cancellation, `too_long` / `undecodable` / `unreadable` errors. |
| API | First Alembic revision (users, sessions, missions, assets), auto-applied on start. Operator accounts: first-run setup (loopback only, no account yet, never in production) or env bootstrap; scrypt hashes; server-side sessions (SHA-256 of an HttpOnly SameSite=Lax cookie, Secure in production); login throttling; `X-Nazar-Request` CSRF guard on all mutations. Owner-scoped missions. Resumable chunked upload with offset checking (409 returns the server offset), size and extension limits, server-generated storage paths, SHA-256. Background validation/preflight on one thread with live progress, cancel, retry after an internal failure, and re-queue on restart. Routes for the preflight report and keyframe/thumbnail images. Readiness gained an `accounts` check. |
| Web | First-run setup / sign-in, sign-out. Mission create/select. Real upload: chunked XHR with byte progress, cancel, automatic retry with backoff, and resume of an interrupted upload by re-choosing the same file. Validation progress by phase. Rejected/failed states with Remove/Retry. Preflight panel: facts, findings by severity, sharpness-over-time chart (flagged bands, keyframe ticks, hover/keyboard readout), keyframe strip, and a keyframe table. |
| Worker | `doctor`, `stages`, and **`run`**: lease → heartbeat thread → stages → events/artifacts, with cancel, lost-lease and failure handling. Stage adapters skip with a reason. `test` = synthetic plumbing pipeline. **`standard` = CPU SfM baseline** (see the Milestone 3 report). Unusable input fails the job with a plain, non-retryable reason (`StageFailed`). |
| Jobs (Milestone 4) | Contracts: `job.queued`, job/stage snapshot, worker lease/heartbeat/event/artifact models. API: jobs, append-only event log with atomic `seq` and idempotency keys, state machine, lease sweeper, worker routes (bearer token + lease header), artifact declare/upload with size+SHA-256 checks, WebSocket stream with REST catch-up, cancel/retry. Web: live job panel, stage table, outputs with downloads, completion sound/vibration (once per watched job, user toggles), point-cloud and camera-path layers with in-place revision swaps, fit/toggles, frame/units labels, mission kept in the URL. Readiness `worker` check is now real (last contact within 60 s). |
| Docs | Research (sample facts corrected; preflight results), architecture (ingestion flow), environment setup (env:init, operator account), README. |

## Full pipeline report (2026-11-04, Windows 11 laptop, RTX 4050 Laptop 6 GB, `--how local`)

`sample1.mp4` (Colosseum orbit, 22 keyframes), no telemetry. Job **completed in 16.2 min**, exit 0. Peak GPU memory **4769 MiB of the 5632 MiB budget** (device-sampled rise over other programs, CUDA context included). No model was skipped.

| Stage | Model | What it contributed | Time | Peak VRAM |
|---|---|---|---|---|
| dynamic_analysis | YOLO11s + SAM 2.1 small | 0 moving objects found (people are a few pixels tall), so nothing masked | 6 s | 690 MiB |
| geometry | Fast3R ViT-L/512 | 22/22 views posed, ~290k confident points | 26 s (114 s cold first load) | 4769 MiB |
| depth_priors | Metric3D-v2 ViT-L | depth + normals for 22 views; 22/22 agree with Fast3R (median rel. error 7.2 %, scale spread 3.4 %) | 54 s | 4324 MiB |
| sensor_optimization | GNSS Sim3 | skipped: no telemetry | — | — |
| dense_reconstruction | fusion | 549,051 points after outliers, 167,425 filled from the depth prior | ~10 s | CPU |
| dense_reconstruction | gsplat 1.5.3 | 276,531 Gaussians, 4000 it; PSNR train 25.84 / held-out 25.90 dB; mean pose shift 0.021 | ~14 min* | 1650 MiB |
| surface_processing | Open3D screened Poisson | 642,369 vertices / 1,284,857 triangles; vertices 56k observed / 499k estimated / 87k inferred (13.6 %) | 22 s | CPU |

\*The splat trained in 282 s on an idle machine; this run shared the CPU with the API test suite.

Artifacts: dynamic masks, depth previews, point cloud (PLY/LAS/GLB), camera trajectory, Gaussian scene PLY + renders, surface mesh GLB, report.

**Scale is not applied from Metric3D.** It implies 1 unit = 6.84 m, but that inherits Fast3R's estimated focal length; on sample1 the mesh comes out about 2.2× smaller than the Colosseum's known size. It is reported under `contributions.depth_priors.scale` with `applied_to_outputs: false`. Metric output still needs GNSS telemetry or a known length.

Fixes found by running it: Metric3D needs an `mmcv.utils` shim (mmengine provides the pieces); runners must use absolute paths (they run with cwd `apps/worker/gpu`); on Windows the CUDA teardown hung a finished runner for ~25 min, so runners now leave via `os._exit` after their result event.

## Telemetry import and GNSS scale (2026-10-28, Windows 11 laptop, CPU only)

### Implemented

- **Parser and validator** (`packages/media/python/nazar_media/telemetry.py`). It reads text only; nothing is executed or fetched.
  - **DJI SRT**, three layouts:
    - `[latitude: ..] [longitude: ..] [rel_alt: .. abs_alt: ..]` (Mini / Air);
    - Mavic 2's `[latitude : ..] [longtitude : ..] [altitude: ..]`;
    - Phantom's `GPS(lon,lat,alt) BAROMETER:..` (longitude first).
  - **Airdata CSV**: uses `isVideo` to find the video start, and converts feet to metres.
  - **Plain CSV**: `time_s`, `latitude`, `longitude`, plus optional `rel_alt_m` / `alt_m`.
  - **Normalisation:**
    - time becomes seconds from the video start;
    - the altitude keeps its kind: *above take-off*, *datum not stated* or *none*, and is never presented as sea-level height;
    - lat/lon go to east-north-up (ENU) metres through WGS84 ECEF, never treated as metres.
  - **Dropped and counted, never repaired:** samples with no fix (0, 0), out-of-range values, non-increasing times, and jumps faster than 45 m/s.
  - **Gaps** over 1.5 s are listed. Positions are interpolated only *within* a gap-free span.
- **API.**
  - `POST /missions/{id}/assets` with `kind: telemetry`: `.srt`/`.csv`, up to 20 MB, one per mission.
  - The upload goes through the same resumable chunked route as video. It is validated as soon as the last byte arrives: either `ready`, or `rejected` with a plain reason.
  - `GET /assets/{id}/telemetry` returns a `TelemetrySummary`: samples, rate, span, extent, height range, gaps, dropped samples, video coverage (once the video is validated) and plain-language findings.
  - A job pins the mission's accepted telemetry when it is created (migration `0003`). The worker fetches it from `inputs/telemetry`. If it was removed in the meantime, the job records that and stays local.
- **Worker `sensor_optimization` stage** (`apps/worker/src/nazar_worker/georef.py`).
  - A Umeyama Sim(3) fit maps the placed keyframes' camera centres to the GNSS positions at their video times. It searches a ±2 s clock offset in 0.1 s steps, with a small preference for zero, then drops gross outliers once and refits.
  - Two separate claims:
    - **Scale** needs ≥ 5 matched keyframes, ≥ 50 % of placed keyframes matched, a baseline ≥ 20 m, and RMS residual ≤ max(3 m, 5 % of baseline).
    - **Orientation** (an ENU georeference) additionally needs altitude, and a path that is not close to a straight line (planarity ≥ 0.15). Otherwise rotation about the line is undetermined.
  - Results:
    - Scale accepted: final artifacts keep their local coordinates, and their frame says `scale_status: metric_sensor` with `metres_per_unit`.
    - The LAS export is written in metres. When orientation was also accepted it is in **UTM** (WGS 84, zone of the first fix, EPSG 326xx/327xx), declared in a GeoKeyDirectory VLR; otherwise local axes. The system id says which.
    - UTM is computed in-house (`apps/worker/src/nazar_worker/utm.py`, Krüger series to n⁴) so the worker needs no PROJ install. Heights keep the telemetry's altitude kind; no vertical datum is claimed.
    - The report gains a `sensor_alignment` section: every number, the thresholds, the ENU origin, and the reasons any claim was rejected.
    - Rejected: the frame stays `unknown`, and the `scale_unknown` warning says the telemetry was supplied but not applied, and why.
  - GTSAM pose-graph fusion is **not** used. One similarity transform is the honest model for one video with consumer GNSS. The stage label now says "Sim(3) GNSS alignment".
- **Web.**
  - A "Flight telemetry · optional" box in the mission panel: attach, see the summary and findings, remove. Rejected files show their reason.
  - Measure reads "… m (GNSS scale, estimate)" when the frame has a sensor scale.
  - The viewer's Units row shows "1 unit = N m, metric (sensor)".

### Verified

- `pnpm check`: 15 of 15 tasks pass.

  | Package | Result |
  |---|---|
  | media | 30 passed (17 new telemetry tests) |
  | worker | 27 passed (7 new alignment tests) |
  | API | 44 passed, 1 skipped (Fast3R, environment-gated) |
  | web | 33 passed |

- Full E2E: 25 passed, 2 skipped. Attach, summarise, remove and reject telemetry now run on desktop and mobile.
- **Through the real pipeline** (`apps/api/tests/test_sensor_pipeline.py`, CPU baseline on `sample1.mp4`):
  - A synthetic track built from the job's own camera path at 10 m/unit was recovered as **10.024 m/unit** (0.24 %).
  - The pipeline correctly **refused to georeference**: sample1's camera moves along a short, nearly straight arc (planarity < 0.15).
  - The LAS came out in local metres, with an extent 10× the local one.
  - A hovering track (span < 20 m) was reported but not applied, and the frame stayed `unknown`.
- **Unit tests on synthetic tracks:**
  - An orbit with 1.5 m GNSS noise gives scale within 3 % plus a georeference.
  - A 0.8 s clock offset is found within 0.1 s.
  - Straight-line flight gives scale only.
  - Rejected: a hover, telemetry from another flight, telemetry with no altitude, and telemetry with no time overlap.
  - ENU checked against known geodesy: 0.001° of latitude at 41.89° N ≈ 111.07 m, and 0.001° of longitude ≈ 82.9 m.

### Performance-accuracy evidence

- **No real flight log was supplied,** so every accuracy figure above comes from **synthetic** telemetry and is labelled so in the tests. It shows the maths and the acceptance rules work. It says nothing about real DJI GNSS accuracy or SRT clock sync. That needs B11.
- The alignment itself costs well under a second: 41 offsets × one SVD on ≤ 96 keyframes.

### Known limitations

- **Scale quality** is bounded by consumer GNSS (a few metres) and by how far the drone flew. The reported `scale_uncertainty_rel_approx` counts fit noise only, not GNSS bias or the time-sync error that remains after the offset search.
- **The cloud stays in its local frame,** with metres-per-unit metadata. Only the LAS export is converted (metres, or UTM when georeferenced). Placement inherits consumer-GNSS error (metres); heights are relative to take-off, not to a vertical datum. Norway/Svalbard UTM zone exceptions are not applied.
- **No altitude** (lat/lon only): no alignment is attempted. A horizontal-only fit would need a known up axis.
- **Formats:** only the SRT layouts and CSV columns above. Other drone logs (e.g. `.DAT`, `.txt` flight records, GPX) are not read.
- **Before validation:** a telemetry file attached before its video is validated shows "coverage known once the video is validated" until then.

### Added after the report: GLB export (same night)

- **Writer.** `apps/worker/src/nazar_worker/glb.py` writes glTF 2.0 binary with numpy only:
  - the final cloud as a POINTS primitive (POSITION float32 with min/max, COLOR_0 normalised RGBA);
  - the camera path as a LINE_STRIP;
  - +Y up, the same axes as NAZAR.
- **Units.** Coordinates are in metres when a GNSS scale was accepted, and unitless otherwise. `asset.extras.units` says which, because glTF viewers assume metres.
- **Job outputs.** Every Reconstruct job now publishes `export-glb` ("Point cloud + camera path (GLB)") next to the LAS.
  - The job snapshot used to keep one artifact per *kind*, so a second export hid the first. It now keys on kind and media type. Previews still replace each other.
- **Verified.**
  - Spec layout and round-trip tests (3 new worker tests).
  - The pipeline tests check that the GLB matches the PLY points and colours, on both CPU and Fast3R (the Fast3R test was run on this laptop's GPU), and that it is in metres with a GNSS scale.
  - three.js's own `GLTFLoader` (already installed for the viewer) parsed a generated file: a `Points` node with 500 coloured points, a `Line` node, the bounds, and the extras.
  - `pnpm check`: 15 of 15 tasks. E2E: 25 passed, 2 skipped.
- **Not verified.** No desktop glTF viewer was opened. See B12.

### Next step

Run B11 with a real DJI clip and its SRT. Then open the georeferenced LAS in a GIS (B13) to confirm it lands on the map.

## Milestone 3 report — Fast3R GPU geometry (2026-10-27, Windows 11 laptop, RTX 4050 Laptop 6 GB)

### Implemented

- **Isolated GPU runtime** (ADR-0004). Fast3R runs in its own Python 3.11 environment (`var/gpu/fast3r-env`, torch 2.7.1+cu126), never inside the worker. Everything is pinned: source commit `33104d4`, weights revision `a2c770b` with the checkpoint SHA-256, and an exact package freeze (`apps/worker/gpu/requirements-fast3r.txt`). `pnpm gpu:setup` installs it and `pnpm gpu:check` verifies it read-only. No token is needed.
- **Runner** (`apps/worker/gpu/fast3r_runner.py`). It streams JSON events: `loaded`, `plan`, `chunk`, `retry`, `done` and `error`. The details:
  - **Pass size.** It sizes passes from free VRAM. Large sequences go in overlapping passes, aligned by a robust Sim(3) fitted on the shared views. Each pass writes a partial result, so the viewer gets previews while it runs.
  - **Out of memory.** An OOM halves the pass size in-process. A driver-level CUDA error is reported as OOM, so the worker restarts the runner smaller.
  - **Loading.** The checkpoint is read with a streaming reader instead of memory-mapping, because Windows charges a copy-on-write map of the 2.5 GB file against commit. The model is built in fp32 on the GPU, then cast to bf16. Peak host commit is reported.
- **Worker adapter** (`nazar_worker/fast3r.py`, `standard.py`).
  - Runs the runner with a clean environment.
  - Converts pointmaps into NAZAR's frame (levelled, local, unknown scale).
  - Filters by confidence, then labels each point *observed* or *estimated* by a stated heuristic.
  - Publishes preview and final clouds plus the camera path.
  - Falls back to the CPU baseline with a reason.
  - Settings come from `.env` (`NAZAR_FAST3R_*`); relative paths resolve against the env file.
- **Web.** Shared provenance helpers (`lib/provenance.ts`). The job tag reads "GPU · Fast3R" or "CPU baseline", and the viewer label states the method. Help text explains both paths.
- **E2E.** Opt-in Fast3R run: `NAZAR_E2E_FAST3R=1`. E2E ports moved to 18765/15174, so they never collide with other local servers.

### Verified

- Worker: ruff, format and strict mypy clean; 15 tests pass. API: mypy clean; `test_standard_pipeline.py` 4/4 with the GPU configured.
- Those 4 tests cover:
  - the CPU baseline;
  - Fast3R dense on sample1 (all keyframes placed, more than 50k points, cameras above the levelled ground, both confidence labels present);
  - a broken GPU runtime, which falls back visibly;
  - an unconfigured runtime.
- Web: lint, typecheck and Vitest pass, including the provenance tests. Playwright real-data spec passes on the CPU path and, with `NAZAR_E2E_FAST3R=1`, on the Fast3R path (twice).
- The streaming checkpoint reader matches safetensors' own loader on all 712 tensors.
- `scripts/setup_fast3r.py --check --hash` passes against the pinned SHA-256.

### Performance-accuracy evidence

Full table and discussion: `docs/research/model-selection.md` ("Fast3R vs the CPU baseline").

| | sample1 (22 kf) | sample2 (78 kf) |
|---|---|---|
| Fast3R points | 267,786 | 205,362 |
| CPU SfM points | 6,340 | 5,040 |
| Fast3R geometry stage | 13.8 s, of which inference ≈ 4.5 s | 50.5 s, 15 passes |
| Fast3R peak VRAM | 3.1 GiB | 3.9 GiB |
| Alignment self-consistency | 1.6 % residual | 7 % median, 17 % max residual |

There is no ground truth, so these numbers are density, speed and self-consistency, **not** metric accuracy. Run-to-run differences are a median of 0.0003 and a maximum of 0.015 (scene units) from GPU nondeterminism.

### Known limitations

- **Pass chaining on a 6 GB GPU.** On long clips (sample2) the chunk alignment drifts: the camera path shows jumps, and every point is labelled *estimated*. A skeleton-anchored alternative was measured and not adopted (see model-selection). A larger GPU (college VDI) is the expected fix, but that is untested.
- **Windows memory.** Under WDDM the runner needs ≈ 5–8 GiB of free system commit. When the laptop runs short, CUDA fails with "unknown error" and the job falls back to the CPU baseline, visibly.
- **No scale or ground truth.** Metric3D-v2 (scale) and telemetry are not integrated.
- **Not interactive yet.** Confidence colouring in the viewer, measurement tools, exports (LAS/GLB) and meshing are not built.
- **Licence.** Fast3R is non-commercial and archived upstream. It is approved for this academic project only.

### Added after the report (same day)

- **Confidence colouring in the viewer.** Final PLY clouds carry a per-point `confidence` class (1 observed, 2 estimated). **Colour by confidence** recolours the cloud (blue/orange; the palette is validated for colour-vision deficiency on the black viewer) and shows a counted legend. API tests check that the classes match the report; E2E checks the legend on both the CPU and Fast3R paths.
- **LAS export.** The final cloud is also published as LAS 1.2, point format 2 (`nazar_worker/las.py`, numpy only):
  - no CRS, because the frame is local; the system identifier says "NAZAR local frame, no CRS";
  - axes converted to +Z up;
  - confidence class in the user-data byte.

  Tests check the header byte offsets against the specification and a round trip, and that the real pipeline's LAS equals its PLY. Opening it in a third-party viewer is pending (B9).
- **Relative measurement.** **Measure** lets you click two cloud points; the viewer draws the segment and reports the distance as "N units (relative; scale unknown)", never in metres, while the scale is unknown. E2E picks two points on both the CPU and Fast3R clouds.
- **Readiness `gpu` check is real.** Workers report their geometry backend (`fast3r` or `cpu-baseline`) on each lease. Readiness shows OK for Fast3R, or explains that the CPU baseline is in use.

### Next step

1. Your checks: B7, B8 and B9 in `PENDING_USER_TESTS.md`.
2. Milestone 1 on the college GPU: `pnpm gpu:check` there, then sample2 with larger passes.
3. On the laptop: telemetry import (gives scale and position), and a GLB export.

## Milestone 3 report — CPU baseline slice (2026-10-27, Windows 11 laptop, CPU only)

### Implemented

- `sfm.py`, a classical incremental SfM:
  - SIFT features, windowed mutual matching, essential-matrix RANSAC, union-find tracks;
  - an initial pair chosen for parallax (planar and rotation-only pairs are rejected);
  - EPnP registration and parallax-gated triangulation;
  - sparse bundle adjustment with an analytic Jacobian, Huber loss and a shared estimated focal length;
  - pruning, normalisation and up-alignment by a RANSAC plane fit.
  - It yields a preview snapshot after initialisation and after every 4 registrations, and stops at safe points when a job is cancelled.
- `baseline.py`, the standard pipeline:
  - keyframes are fetched from the API;
  - `preview-{n}` point clouds stream while geometry runs;
  - the final cloud is a statistically filtered one, published with the camera path (poses + estimated focal length);
  - each point is labelled observed or estimated;
  - a processing report (`nazar.processing-report/1`) records the method, statistics, confidence rule, coordinate frame, timings, resources, skip reasons and limitations.
  - Depth priors, dynamics and surface are skipped with reasons; sensor fusion is skipped when there is no telemetry.
- Web:
  - **Reconstruct** is enabled and starts the standard pipeline.
  - The job tag reads "CPU baseline". The viewer tag comes from artifact provenance ("CPU baseline · sparse SfM").
  - Stage copy no longer promises outputs this run cannot make (no GLB/LAS/Gaussian claims).
- Dependencies (worker only): `opencv-python-headless` 5.0 (Apache-2.0), `scipy` 1.18 (BSD-3), plus `scipy-stubs` for dev. No model weights.
- `docs/research/model-selection.md`: licence and hardware matrix for Fast3R, VGGT, MASt3R and COLMAP, verified on the official repositories. Nothing was downloaded.

### Verified

- Worker pytest, 14 tests:
  - bundle adjustment recovers a perturbed synthetic scene and an 8 % focal error back to the injected 0.3 px noise;
  - triangulation refuses zero-parallax pairs;
  - up-alignment levels a tilted ground plane;
  - the array PLY writer is byte-identical to the tuple writer;
  - **real `sample1.mp4` → real preflight → reconstruction**: previews grow monotonically, ≥ 80 % of keyframes are registered, ≥ 3000 points, mean reprojection < 1.5 px, and the cameras sit above the ground.
- The analytic BA Jacobian matches finite differences to 1e-8.
- API test: a standard job on undecodable keyframes fails with "Keyframe 1 could not be decoded.", is not retryable, and publishes no artifacts.
- Playwright **real-data E2E** (desktop):
  - upload sample1, press Reconstruct, and a real preview cloud (> 500 points) renders *while the job is still running*;
  - the job completes and the final cloud has > 3000 points;
  - geometry shows done and depth priors show skipped;
  - the unknown-scale warning is shown;
  - the downloaded report has ≥ 18 registered keyframes, mean reprojection < 1.5 px, `scale_status: unknown`, and a point count equal to the cloud on screen;
  - no console errors.
- E2E now runs two real worker processes (test and standard), as in deployment.
- **Bug found and fixed by the fuller E2E:** a Cancel that landed between the worker's cancel check and its next `stage.started` made the job end *failed* instead of *cancelled*. The worker now asks the API when a stage start is refused, and reports a cancellation when that is the reason. A deterministic API regression test injects the cancel at exactly that point; it fails without the fix.
- Totals: `pnpm check` 15/15 (API 35, worker 14, media 13, contracts 7, web 24). Playwright 23 passed, 2 skipped (desktop-only by design).
- The final screenshot was reviewed: the Colosseum's oval ring is visible.

### Performance-accuracy evidence

sample1 (22 keyframes, 640×360):

| Measure | Result |
|---|---|
| Registered keyframes | 22 / 22 |
| Points (raw / after outlier filter) | 6454 / 6340 |
| Observed / estimated points | 4518 / 1822 |
| Mean reprojection | 0.63 px |
| BA RMS (Huber) | 1.06 px |
| Estimated focal length | ≈ 481 px |
| Standalone timing | first preview ≈ 8–13 s, total ≈ 18 s |
| Timing inside the busy E2E run (browser + API + 2 workers on the same laptop) | first preview 18.5 s, geometry 34.3 s |

There is no ground truth for the sample, so absolute accuracy is **not measured**. The reprojection error measures internal consistency only.

### Known limitations

- Sparse points only. There is no dense surface, mesh or Gaussian scene; textureless areas are empty.
- Unknown scale and a local frame. Up is estimated from the dominant plane.
- The focal length is not calibrated, and no lens distortion is modelled.
- Footage that only rotates or hovers has no parallax: the job fails with a reason instead of inventing geometry.
- sample2 (78 keyframes) has not yet been run end to end through the UI (B7).
- The GPU models from the brief are not integrated (C1 licence approval + Milestone 1 GPU access).

### Next step

1. Get C1 and C3 answers, and access to the college GPU.
2. Write a Fast3R adapter for the `geometry` stage behind the same contract. The CPU baseline stays as the fallback.
3. Meanwhile, on CPU: telemetry import, measurement tools and confidence colouring in the viewer.

## Local or college GPU per job (2026-11-02)

### Implemented
- **Run on: Local GPU / College GPU** switch above Reconstruct. It shows whether the chosen worker is online and which method it will use, and is remembered per browser. The job panel tags each job with its GPU; a queued college job says it is waiting for the college worker.
- Contracts: `ComputeTarget`; `JobCreate.compute`, `Job.compute`, `WorkerLeaseRequest.location`, `ReadinessResponse.workers` (`WorkerPresence`). Migration `0004_job_compute` (existing jobs become `local`).
- API: leases only match the worker's location; presence and geometry tracked per location from the `X-Nazar-Worker-Location` header (unknown values rejected with 400). The `worker` / `gpu` readiness checks summarise both.
- Worker: `run --location local|college` (env `NAZAR_WORKER_LOCATION`); `bundle export --location` (default college); bundles carry the location to import.

### Verified
- `apps/api/tests/test_compute.py` (5): default local; a college job is ignored by the local worker and run by the college one; retry keeps the location; per-location readiness; bad header rejected; a college job through a bundle.
- Web unit tests for the presence text; E2E: choose College GPU, start a job, the local worker leaves it queued and labelled, cancel, choice remembered (desktop + mobile).
- `pnpm check`: 15 of 15 tasks (API 53 passed, 1 skipped; web 35).

### Known limitations
- No automatic fallback between GPUs, by design: a college job waits until a college worker connects (or is cancelled and restarted on the local GPU).
- Not yet exercised with a real college worker (todoCollege.md step 7).

## Offline bundle mode (2026-10-28)

### Implemented
- `nazar-worker bundle export | run | import` (`apps/worker/src/nazar_worker/bundle.py`). ADR-0003's fallback for a VDI with no route to the API. Flow and safeguards are in the ADR.
- Contract: optional `job_id` on `WorkerLeaseRequest`; the API's `lease_next` leases only that job when it is given.
- `WorkerApi` protocol: `JobRun` runs unchanged against the HTTP client or the offline `BundleClient`.

### Verified
- `apps/api/tests/test_bundle.py` (4 tests, in-process API):
  - test-pipeline round trip: job completes on attempt 1 with the same artifacts as an online run, ordered events, real stage durations and the `offline_bundle` warning; a second import is refused;
  - import refused while the export lease is held, and for a result from a different video;
  - a corrupted artifact is refused before any event is sent; the job stays queued;
  - REAL DATA: sample1 through the actual CLI `bundle run` (CPU baseline), imported with LAS and GLB exports.
- `pnpm check`: 15 of 15 tasks (API 48 passed, 1 skipped).

### Known limitations
- Not yet run on the college VDI (todoCollege.md step 5/6).
- No live preview while offline; the whole job appears at import.
- If an import is interrupted, wait for its lease to lapse (60 s) and import again; it then counts as attempt 2.

### Next step
Use it at college if outbound HTTPS is blocked (A3).

## Georeferenced LAS in UTM (2026-10-28)

### Implemented
- `utm.py`: ENU → geodetic (iterative, WGS84) → UTM (Krüger series to n⁴), plus zone and EPSG selection.
- `las.py`: optional GeoKeyDirectory VLR (`LASF_Projection`, record 34735) declaring ProjectedCSTypeGeoKey, metre units; `read_las` reports the EPSG code back.
- `Alignment.origin` and `Alignment.to_utm`; the georeferenced LAS branch now writes UTM with the VLR, and the report records `crs`.
- Clock-offset search now needs a 5 cm absolute gain (as well as 5 %) before it prefers a non-zero offset. On a constant-speed orbit a time shift is the same as a rotation, so float noise could otherwise pick an offset and rotate the whole georeference.

### Verified
- UTM matches PROJ (pyproj 3.7.1, run once in a throwaway environment; not a project dependency) to within 1 mm at six points: Rome, Sydney (south), Reykjavik, the central meridian, the equator and a zone edge.
- ENU → geodetic → ENU round trip under 1 µm.
- GeoKey VLR round-trips through `write_las` / `read_las`, and the point offset accounts for it.
- On a SYNTHETIC orbit, georeferenced keyframes land on the track's own UTM coordinates within 0.25 m. The residual comes from the test helper's flat-earth lat/lon, not the code.
- `pnpm check`: 15 of 15 tasks (worker 41 passed).

### Performance-accuracy evidence
- Projection error versus PROJ: < 1 mm. The real accuracy limit is the GNSS track (metres) and the alignment RMS shown in the report.

### Known limitations
- No vertical datum: heights are the telemetry's (usually above take-off).
- Norway/Svalbard zone exceptions not applied (the standard zone is still valid).
- Not yet opened in a real GIS (B13).

### Next step
B11 (real DJI SRT) then B13 (open the LAS in QGIS/CloudCompare).

## Milestone 4 report (2026-10-27, Windows 11 laptop, CPU only)

### Implemented

- Job lifecycle (`created → queued → stages → completed | failed | cancelled`), derived only from the append-only event log.
- Pull-based worker protocol from ADR-0003: lease, heartbeat, events, inputs, artifacts, lease expiry and re-queue.
- Live browser updates over WebSocket, with REST catch-up and backoff reconnect.
- Progressive viewer: preview revisions, then the final cloud and camera path, without resetting the camera.
- Completion feedback, outputs inventory, cancel and retry.
- `pnpm worker` (test pipeline) and real `worker` readiness.

### Verified

- `pnpm check`: 15/15 Turbo tasks pass. That covers pytest (contracts 7, worker 8, media 13, API 34), vitest 24, ruff, strict mypy, ESLint, Prettier, `tsc` and contract drift.
- API tests cover:
  - a full run through the real `run_loop`;
  - WebSocket replay from a cursor, plus origin and anonymous rejection;
  - token checks, event idempotency and state-machine rejections;
  - an artifact checksum mismatch;
  - lost lease → re-queue;
  - cancel and retry;
  - the `standard` pipeline failing as non-retryable;
  - owner isolation.
- Playwright: 22 passed and 1 skipped (desktop-only by design), on desktop and Pixel 7. The suite runs the real API, a real worker process (with a random per-run token that is never written to disk) and Vite. The job tests check:
  - stages stream;
  - a preview cloud is rendered **before** the job completes;
  - completion shows the final cloud;
  - the sensor stage is skipped with "No telemetry";
  - the synthetic warning is shown;
  - the downloaded PLY is valid;
  - layer toggles and Fit work;
  - a reload during a running job resumes it, then cancel and retry work.
- Screenshots of the completed job were reviewed at desktop and mobile widths.

### Performance-accuracy evidence

- **Accuracy:** none. The geometry is synthetic, so no accuracy claim is made.
- **Plumbing timings** (test pipeline, `--step-delay 0.8`):
  - the run takes about 10 s;
  - preview revisions have about 13k points;
  - the final cloud has about 26k points (≈380 KB binary PLY);
  - an event reaches the browser within about 0.25 s, which is the server poll interval.

### Known limitations

- No real reconstruction yet (Milestone 3).
- The WebSocket polls the database every 0.25 s; there is no pub/sub, which is fine for one operator.
- Leases live in SQLite. With more than one API process you would need PostgreSQL and row locks.
- Artifacts are stored on the API host's disk. The worker reads inputs over HTTP; there is no shared storage.
- Completion sound and vibration, network-drop reconnect and VDI connectivity have not been checked by a person (see `PENDING_USER_TESTS.md`).
- The GPU readiness check still says "not built yet".

### Next step

Milestone 3 research matrix: candidate geometry models with licence, weight terms and VRAM needs. Nothing gets downloaded until you approve it (C1 in `PENDING_USER_TESTS.md`).

## Milestone 2 evidence (actually run, 2026-10-26, Windows 11 laptop, CPU only)

1. **`pnpm check`:** 15/15 Turbo tasks pass.
   - pytest: contracts 6, worker 5, media 13, API 23.
   - vitest: 16, including the upload resume/retry state machine and chart downsampling.
   - ruff, strict mypy, ESLint, Prettier, `tsc`, contract drift.
2. **API tests cover:**
   - setup closes after the first account and is refused in production;
   - bootstrap login, wrong password, throttling (429), CSRF header (403), short password (422);
   - missions require sign-in;
   - wrong type / oversize rejected; stale and over-long chunks;
   - a corrupt `.mp4` is rejected with a reason and can be replaced;
   - **`sample1.mp4` upload → `ready` → preflight with >500 frames and ≥8 keyframes → JPEG thumbnail served;**
   - cancel during validation cleans up.
3. **Playwright:** 18 passed, 1 skipped (keyboard test is desktop-only by design), desktop and Pixel 7, real API + Vite.
   - A `setup` project creates the operator through the first-run UI in a fresh per-run database.
   - Sample upload through the UI → preflight panel, chart, ≥8 loaded keyframe thumbnails, no console errors.
   - Non-video rejected in the browser; corrupt video rejected by the server and removable.
   - Sign in (wrong then right password) and sign out.
   - Screenshots of the preflight screen were reviewed at both sizes. The chart now draws at real pixel width so its labels stay 10 px.
4. **Preflight numbers** (see `docs/research/README.md`): sample1 22 keyframes in ≈1.6 s; sample2 78 keyframes in ≈3 s with a real repeated-frames warning (289 of 1033).
5. **Secrets:** `.env` stays ignored. `playwright/.auth/` (E2E session cookie) is ignored. No secret values appear in code, tests, logs or this file.

## Known gaps and warnings

- **Telemetry** is tested on synthetic tracks and on synthetic telemetry through the real pipeline only. No real flight log has been tried yet (B11).
- **Under load, E2E timing grows.** Several Chromium instances with software WebGL plus server-side decoding starved the laptop at default parallelism (false timeouts). Local workers are capped at 3; the validation tests have explicit time budgets.
- **Preflight thresholds** were calibrated on the two supplied clips only. Treat them as hypotheses for other footage.
- **Validation runs in the API process** (one background thread). This is fine for one operator. GPU work will use the pull-based worker (ADR-0003).
- **Login throttling is in memory.** It resets on restart and is per process.
- **First-run setup trusts loopback.** Through the Vite dev proxy every request is loopback, so keep the dev server bound to 127.0.0.1 (the default) and use env bootstrap for anything exposed.
- `pytest` shows a harmless `StarletteDeprecationWarning` (the test client wants `httpx2`; not yet vetted).
- Real reconstruction is the **full model pipeline** (GPU) or the **CPU SfM baseline** (sparse). The splat is a gsplat optimisation started from the fused cloud (InstantSplat-style), not the InstantSplat code.
- **Moving-object masking is untested on real moving objects:** neither sample clip has any large enough to detect.
- **The web viewer does not show the mesh or the splat yet** (they are downloadable artifacts; backend first).
- **Metric3D scale is reported, not applied** (see the full pipeline report).
- The college GPU has not been accessed. The `clg` profile's weights (Metric3D ViT-giant2 ~5.5 GB, SAM 2.1 large ~0.9 GB, YOLO11x ~0.1 GB) are not downloaded yet.

## Next action

0. **Your checks:** [`PENDING_USER_TESTS.md`](PENDING_USER_TESTS.md).
1. **Milestone 1** (needs the user): start a SyncVDI session, run `nazar-worker doctor` and read-only network checks there, record the results, and choose ADR-0003's online or bundle mode.
2. **Full pipeline on the college GPU:** `python scripts/setup_gpu.py --profile clg`, then `scripts/test_full_pipeline.py --how clg` (see `todoCollege.md` step 7); compare with the laptop numbers above.
3. **Viewer:** mesh layer and splat viewer in the web app.
4. **Telemetry with real data** (B11) and the georeferenced LAS in a GIS (B13).

## Resume instructions

`pnpm install && uv sync --all-packages && pnpm env:init && pnpm check && pnpm test:e2e` (optionally `pnpm gpu:check`), then continue from "Next action". To try it by hand: `pnpm dev`, then `pnpm worker` in a second terminal.
