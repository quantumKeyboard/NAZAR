# NAZAR

Single-pass drone video → progressively revealed 3D reconstruction.

> **Status: frontend connected to the full pipeline on `feature/frontend-pipeline`.** Upload a video in the browser, leave the frame count on Automatic (about one frame per second, at least 24) or choose frames per second, and the local GPU worker runs YOLO11 + SAM 2 → Fast3R → Metric3D-v2 → fusion → Gaussian splat → a closed height-map mesh textured from your full-resolution frames while the page shows the current step, the model, live counts and an estimated time left. The viewer shows the photoreal splat, the solid mesh and the points, and lists what each model contributed. See [`docs/DEMO.md`](docs/DEMO.md) for the click path and [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Layout

```text
apps/web            React 19 + TypeScript + Vite + React Three Fiber + Tailwind v4 + Zustand
apps/api            FastAPI service (Python 3.12, uv) — auth, missions, uploads, preflight
apps/worker         Reconstruction worker CLI (doctor, stage plan, job loop; CPU SfM baseline + synthetic test pipeline)
packages/contracts  Pydantic contracts → JSON Schema → generated TypeScript
packages/media      FFmpeg probe/decode, frame quality metrics, keyframe selection (CPU)
infra/              Optional docker-compose (PostGIS, Redis, MinIO)
docs/               Project brief, design guide, VDI guide, samples + our ADRs/notes
```

## Quick start

```bash
pnpm install
uv sync --all-packages
pnpm env:init               # creates .env and generates local machine secrets (prints names only)
pnpm dev                    # web http://127.0.0.1:5173  ·  API http://127.0.0.1:8000/docs
pnpm worker                 # second terminal: worker for the CPU baseline (Reconstruct) and the test pipeline
                            # first visit: create the operator account in the web app
```

On Windows, after the setup lines, double-click instead of `pnpm dev` + `pnpm worker`:

- `run-local.bat`: API, web and a worker on this laptop's GPU (Fast3R from `var/gpu` when set
  up); opens the browser with **Run on → Local GPU**.
- `run-college.bat`: API and web only, with **Run on → College GPU**. It asks for _Online_ or
  _Bundle_. Online opens the API to the campus network after a confirmation; in the VDI run
  `bash scripts/run-college-worker.sh <laptop IP>`. Bundle keeps the API private and walks
  through export, portal Storage and import. See `docs/todoCollege.md`.

## Checks

```bash
pnpm check                  # lint + typecheck + unit tests + contract drift (TS and Python)
pnpm test:e2e               # Playwright: real API + web, desktop and mobile
uv run --package nazar-worker nazar-worker doctor   # read-only GPU/disk/toolchain report
```

## Documentation

- Source documents (unchanged): [project brief](docs/Single_Pass_Drone_3D_Reconstruction_Detailed_Project_Documentation.md), [design guide](docs/DESIGN.md), [SyncVDI guide](docs/SyncAI-VDI%20User%20Guide.pdf), [build prompt](docs/Claude_Prompt_Single_Pass_Drone_3D_Reconstruction.md), sample videos in `docs/sample/video/`
- [Full Pipeline Completion Roadmap](docs/pipeline-completion/README.md) — step-by-step guides for surface meshing (Open3D), depth priors (Metric3D-v2), dynamic masks (YOLO+SAM2) and Gaussian splatting
- [Demo guide](docs/DEMO.md) · [Implementation status](docs/IMPLEMENTATION_STATUS.md) · [Environment setup](docs/ENVIRONMENT_SETUP.md)
- [Architecture](docs/architecture/overview.md) · [Testing](docs/testing/README.md) · [Research](docs/research/README.md)
- Decisions: [ADR-0001 tooling](docs/decisions/ADR-0001-monorepo-tooling.md) · [ADR-0002 local-first infra](docs/decisions/ADR-0002-local-first-infrastructure.md) · [ADR-0003 pull-based GPU worker](docs/decisions/ADR-0003-pull-based-gpu-worker.md) · [ADR-0004 isolated model envs](docs/decisions/ADR-0004-isolated-model-environments.md) · [ADR-0005 design system & fonts](docs/decisions/ADR-0005-design-system-fonts.md)
