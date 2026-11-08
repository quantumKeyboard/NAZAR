"""3D Gaussian splat scene from the fused cloud + Fast3R cameras (gsplat, Apache-2.0).

Runs INSIDE the splat environment (torch + gsplat; see `docs/ENVIRONMENT_SETUP.md`). The
approach follows InstantSplat: the Gaussians start on the dense learned geometry (here the
fused Fast3R + Metric3D cloud, not DUSt3R) and are optimised *together with the camera
poses* against the keyframes for a short schedule, without adaptive densification. Moving
objects (SAM 2 masks) are excluded from the photometric loss.

    python splat_runner.py --frames DIR --cameras cameras.json --points init.npz --out DIR
        [--masks DIR] [--iterations 4000] [--max-gaussians 400000] [--sh-degree 1]
        [--budget-mib 5632]

cameras.json: {"cameras": [{"image": "0000.jpg", "c2w": 4x4 (NAZAR world, OpenCV camera
axes), "fx", "fy", "cx", "cy", "width", "height", "split": "train"|"test"}]}
Writes scene.ply (standard 3DGS layout), poses.json (refined), renders/*.jpg (held-out views:
photo | render) and metrics in the `done` event. Held-out views are scored after a short
pose-only alignment with the Gaussians frozen (InstantSplat's evaluation protocol).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from _common import Timer, budget, emit, is_oom, log, peak, run_main

SH_C0 = 0.28209479177387814


def ssim(a: Any, b: Any) -> Any:
    """Mean SSIM of two (1,3,H,W) images in [0, 1], 11-px Gaussian window."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    w = g.expand(3, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=3), F.conv2d(b, w, padding=5, groups=3)
    va = F.conv2d(a * a, w, padding=5, groups=3) - mu_a**2
    vb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b**2
    cov = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return s.mean()


def rodrigues(w: Any) -> Any:
    """Axis-angle (C,3) -> rotation (C,3,3), differentiable, safe at zero."""
    import torch

    theta = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / theta
    kx = torch.zeros(w.shape[0], 3, 3, device=w.device)
    kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
    kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, device=w.device) + s * kx + (1 - c) * (kx @ kx)


class Scene:
    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, scene_scale: float):
        import torch
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(xyz).query(xyz, k=4)
        scale = np.log(np.clip(dist[:, 1:].mean(1), 1e-6, None))
        n, k = len(xyz), (sh_degree + 1) ** 2
        dev = "cuda"
        f32 = torch.float32
        self.params = {
            "means": torch.tensor(xyz, dtype=f32, device=dev),
            "scales": torch.tensor(np.repeat(scale[:, None], 3, 1), dtype=f32, device=dev),
            "quats": torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=f32, device=dev),
            "opacities": torch.full((n,), math.log(0.1 / 0.9), dtype=f32, device=dev),
            "sh0": torch.tensor((rgb / 255.0 - 0.5) / SH_C0, dtype=f32, device=dev)[:, None],
            "shN": torch.zeros(n, k - 1, 3, dtype=f32, device=dev),
        }
        for p in self.params.values():
            p.requires_grad_(True)
        lrs = {
            "means": 1.6e-4 * scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }
        self.opt = {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15) for k, p in self.params.items()}
        self.mean_lr0 = lrs["means"]
        self.sh_degree = sh_degree

    def render(self, viewmats: Any, ks: Any, w: int, h: int, sh: int) -> Any:
        import torch
        from gsplat import rasterization

        p = self.params
        colors, _, _ = rasterization(
            means=p["means"],
            quats=torch.nn.functional.normalize(p["quats"], dim=-1),
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]),
            colors=torch.cat([p["sh0"], p["shN"]], 1),
            viewmats=viewmats,
            Ks=ks,
            width=w,
            height=h,
            sh_degree=sh,
            packed=True,
        )
        return colors[..., :3].clamp(0, 1)

    def keep(self, mask: Any) -> None:
        """Drop Gaussians (and their optimiser state) where mask is False."""
        import torch

        for name, p in list(self.params.items()):
            new = torch.nn.Parameter(p.detach()[mask].clone())
            opt = self.opt[name]
            state = opt.state.pop(p, None)
            opt.param_groups[0]["params"] = [new]
            if state:
                state = {
                    k: v[mask] if torch.is_tensor(v) and v.dim() else v for k, v in state.items()
                }
                opt.state[new] = state
            self.params[name] = new

    def save_ply(self, path: Path) -> int:
        p = {k: v.detach().cpu().numpy() for k, v in self.params.items()}
        n = len(p["means"])
        rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)
        cols = [("x", p["means"][:, 0]), ("y", p["means"][:, 1]), ("z", p["means"][:, 2])]
        cols += [(f"n{a}", np.zeros(n, np.float32)) for a in "xyz"]
        cols += [(f"f_dc_{i}", p["sh0"][:, 0, i]) for i in range(3)]
        cols += [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])]
        cols += [("opacity", p["opacities"])]
        cols += [(f"scale_{i}", p["scales"][:, i]) for i in range(3)]
        cols += [(f"rot_{i}", p["quats"][:, i]) for i in range(4)]
        rec = np.empty(n, dtype=[(name, "<f4") for name, _ in cols])
        for name, values in cols:
            rec[name] = values
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header += [f"property float {name}" for name, _ in cols] + ["end_header"]
        path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rec.tobytes())
        return n


def psnr(a: Any, b: Any) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--max-gaussians", type=int, default=400_000)
    ap.add_argument("--sh-degree", type=int, default=1)
    ap.add_argument("--budget-mib", type=int, default=0)
    ap.add_argument(
        "--max-side",
        type=int,
        default=1920,
        help="Training images are shrunk (intrinsics scaled) to at most this long side, so a "
        "4K clip does not exhaust GPU memory; 1080p footage is used at full size.",
    )
    args = ap.parse_args()

    import torch

    timer = Timer()
    budget(args.budget_mib)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cams = json.loads(args.cameras.read_text(encoding="utf-8"))["cameras"]
    images, masks = [], []
    for cam in cams:
        img = cv2.imread(str(args.frames / cam["image"]), cv2.IMREAD_COLOR)
        shrink = min(1.0, args.max_side / max(img.shape[:2]))
        if shrink < 1.0:
            img = cv2.resize(img, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            for k in ("fx", "fy", "cx", "cy"):
                cam[k] = cam[k] * shrink
        images.append(
            torch.tensor(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device="cuda"
            )
        )
        m = None
        if args.masks is not None and (args.masks / cam["image"].replace(".jpg", ".png")).is_file():
            mm = cv2.imread(str(args.masks / cam["image"].replace(".jpg", ".png")), 0)
            if mm.shape[:2] != img.shape[:2]:
                mm = cv2.resize(mm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            m = torch.tensor(mm < 128, dtype=torch.float32, device="cuda")[..., None]
        masks.append(m)
    c2w = torch.tensor(np.array([c["c2w"] for c in cams]), dtype=torch.float32, device="cuda")
    ks = torch.tensor(
        np.array([[[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]] for c in cams]),
        dtype=torch.float32,
        device="cuda",
    )
    train = [i for i, c in enumerate(cams) if c["split"] == "train"]
    test = [i for i, c in enumerate(cams) if c["split"] == "test"]
    centres = c2w[:, :3, 3]
    scene_scale = float((centres - centres.mean(0)).norm(dim=1).max()) * 1.1 or 1.0

    data = np.load(args.points)
    xyz, rgb = data["points"].astype(np.float32), data["colors"].astype(np.float32)
    if len(xyz) > args.max_gaussians:
        pick = rng.choice(len(xyz), args.max_gaussians, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    scene = Scene(xyz, rgb, args.sh_degree, scene_scale)
    pose_delta = torch.zeros(len(cams), 6, device="cuda", requires_grad=True)
    pose_opt = torch.optim.Adam([pose_delta], lr=1e-4)
    emit(
        event="plan",
        gaussians=len(xyz),
        train_views=len(train),
        test_views=len(test),
        iterations=args.iterations,
    )

    def viewmat(i: list[int]) -> Any:
        d = pose_delta[i]
        rot = rodrigues(d[:, :3]) @ c2w[i, :3, :3]
        t = c2w[i, :3, 3] + d[:, 3:] * scene_scale
        w2c = torch.zeros(len(i), 4, 4, device="cuda")
        w2c[:, :3, :3] = rot.transpose(1, 2)
        w2c[:, :3, 3] = -(rot.transpose(1, 2) @ t[..., None])[..., 0]
        w2c[:, 3, 3] = 1
        return w2c

    def loss_for(i: int, sh: int) -> Any:
        img = images[i]
        h, w = img.shape[:2]
        out = scene.render(viewmat([i]), ks[[i]], w, h, sh)[0]
        m = masks[i] if masks[i] is not None else torch.ones_like(img[..., :1])
        l1 = ((out - img).abs() * m).sum() / (m.sum() * 3 + 1e-6)
        s = ssim((out * m).permute(2, 0, 1)[None], (img * m).permute(2, 0, 1)[None])
        return 0.8 * l1 + 0.2 * (1 - s), out

    recent: list[float] = []
    it, oom_retries = 0, 0
    while it < args.iterations:
        try:
            sh = min(args.sh_degree, it // 1000)
            i = train[int(rng.integers(len(train)))]
            loss, out = loss_for(i, sh)
            loss.backward()
            frac = it / max(args.iterations, 1)
            for g in scene.opt["means"].param_groups:
                g["lr"] = scene.mean_lr0 * (0.01**frac)
            for opt in scene.opt.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            if it > 200:  # let the colours settle before the poses move
                pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
            if it >= args.iterations - 200:
                recent.append(psnr(out.detach(), images[i]))
            if it % 500 == 499 and len(scene.params["means"]) > 1000:
                with torch.no_grad():
                    alive = torch.sigmoid(scene.params["opacities"]) > 0.005
                if not bool(alive.all()):
                    scene.keep(alive)
            it += 1
            if it % 250 == 0 or it == args.iterations:
                emit(
                    event="progress",
                    step="train",
                    done=it,
                    total=args.iterations,
                    gaussians=len(scene.params["means"]),
                    loss=round(float(loss), 5),
                    elapsed_s=timer.s(),
                )
        except Exception as exc:
            if not is_oom(exc) or oom_retries >= 3:
                raise
            oom_retries += 1
            for opt in scene.opt.values():
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            with torch.no_grad():
                order = torch.argsort(scene.params["opacities"], descending=True)
                mask = torch.zeros_like(scene.params["opacities"], dtype=torch.bool)
                mask[order[: len(order) // 2]] = True
            scene.keep(mask)
            torch.cuda.empty_cache()
            emit(
                event="retry",
                reason="out of GPU memory within the budget",
                gaussians=len(scene.params["means"]),
            )
    train_s = timer.s()

    # Held-out views: align only their pose (Gaussians frozen), then score.
    for p in scene.params.values():
        p.requires_grad_(False)
    (args.out / "renders").mkdir(parents=True, exist_ok=True)
    test_psnr = []
    for i in test:
        opt = torch.optim.Adam([pose_delta], lr=1e-4)
        for _ in range(200):
            loss, _ = loss_for(i, args.sh_degree)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, out = loss_for(i, args.sh_degree)
        test_psnr.append(psnr(out, images[i]))
        side = torch.cat([images[i], out], 1).cpu().numpy()
        cv2.imwrite(
            str(args.out / "renders" / cams[i]["image"]),
            cv2.cvtColor((side * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

    n = scene.save_ply(args.out / "scene.ply")
    with torch.no_grad():
        refined = viewmat(list(range(len(cams)))).inverse().cpu().numpy()
        drift = pose_delta[:, 3:].norm(dim=1).mul(scene_scale).cpu().numpy()
    (args.out / "poses.json").write_text(
        json.dumps({"c2w": refined.round(6).tolist()}), encoding="utf-8"
    )
    emit(
        event="done",
        gaussians=n,
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        train_views=len(train),
        test_views=len(test),
        train_psnr_db=round(float(np.mean(recent)), 2) if recent else None,
        test_psnr_db=round(float(np.mean(test_psnr)), 2) if test_psnr else None,
        pose_refinement_mean_shift=round(float(drift.mean()), 5),
        oom_retries=oom_retries,
        timings={"train_s": train_s, "total_s": timer.s()},
        **peak(),
    )
    log(f"splat: {n} Gaussians, test PSNR {np.mean(test_psnr) if test_psnr else 'n/a'}")
    return 0


if __name__ == "__main__":
    run_main(main)
