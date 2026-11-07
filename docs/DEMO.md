# Demo guide: video in, explainable 3D model out (local GPU)

What you can show in about ten minutes, on this laptop (RTX 4050, 6 GB). Everything runs
locally; nothing leaves the machine.

## Start

1. One-time: `pnpm install`, `uv sync --all-packages`, `pnpm env:init`, and `pnpm gpu:setup`
   if `var/gpu` does not hold the models yet (about 7 GB; already done on this laptop).
2. Double-click `run-local.bat` (API + web + local-GPU worker). The browser opens with
   **Run on → Local GPU**.
3. First visit only: create the operator account.

## Click path

| Step | What to do | What to point out |
| --- | --- | --- |
| 1 | Create a mission, upload `docs/sample/video/sample1.mp4` | Preflight: duration, usable frames, keyframes, warnings (low resolution, no telemetry) |
| 2 | Under **Frames to use** keep **Automatic (recommended)**, or switch to **Choose frames per second** (slider, number box, presets) | Automatic: "24 frames from a 6.6 s video" (about one per second, never fewer than 24); the estimate and the **models on this worker** (green ticks) update live. In manual mode, above the frame cap the app says it lowered the rate; below 3 frames it asks for a higher one |
| 3 | **Reconstruct** | Step _N of 7_, the model that is running, counts ("frame 12 / 23", "iteration 2,000 / 4,000"), elapsed clock, **"about A–B min left"**, and the basis of that estimate |
| 4 | Watch the pipeline cards | Each stage: forecast vs. real time, GPU memory vs. the 5.5 GB budget, and a plain reason for any skipped stage (for example no GPS telemetry) |
| 5 | When it finishes | **What each model contributed** (masks, camera poses, depth, fusion, splat PSNR, mesh share inferred); peak GPU memory within budget |
| 6 | Viewer | Default is the **photoreal splat**. **Solid mesh** is a closed block textured from your frames (vertex colours if no texture could be made); toggle it and **Points**, **Colour by confidence**, **Measure** (relative units, since there is no GPS), **Camera path**, **Fit** |
| 7 | Outputs list | Download the splat PLY, mesh GLB, point cloud (PLY/LAS/GLB), camera path and report |

## Numbers from the recorded run (2026-11-05)

`sample1.mp4`, 1 fps, 23 frames, local profile, run through the browser:

- Finished in **8.4 min**; peak GPU **4.6 GB of 5.5 GB**; all six models ran; 574,475 points,
  271,045 Gaussians, 1,402,917 mesh triangles, console clean.
- The pre-run estimate said "about 4–5 min", which was low. The cause was a calibration
  history polluted by runs without a splat; it is fixed (see the status document), but the
  corrected estimate has not been re-timed on a second full run.

## Honest limits to mention

- Geometry is relative-scale: Metric3D's implied scale (1 unit ≈ 6.8 m here) is reported, not
  applied. A metric result needs GPS telemetry.
- The mesh is vertex-coloured and bumpy in places; about 13 % of its surface is inferred, not
  seen. The splat is the closest match to the video.
- Fast3R weights are for non-commercial academic use.
- The college profile (larger models, more frames, 128-frame cap) is not downloaded or timed
  here; its time forecast is an unmeasured guess.
- The local frame cap (64) has not been stress-tested at high fps on the 6 GB GPU.
