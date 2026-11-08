"""Surface reconstruction: fused point cloud -> coloured triangle mesh (Open3D, CPU only).

Runs inside the main GPU environment because that is where Open3D is installed; it uses no
GPU. Steps:

1. Normals: Metric3D-v2 normals where the fusion supplied them; elsewhere estimated from the
   neighbourhood and flipped to face *the camera that observed each point* (one mean camera
   would flip walls on an orbit).
2. Screened Poisson reconstruction.
3. Trim what the input does not support: low Poisson density, and any vertex farther than
   `--max-gap` point spacings from every input point (Poisson's "bubbles").
4. Clean-up, small disconnected pieces dropped, then Taubin smoothing (volume-preserving).
5. Per-vertex colour (nearest input points) and confidence class: observed / estimated from
   the nearest input point's class when the vertex lies on the data, inferred when it bridges
   a gap between data (more than `--inferred-gap` spacings from any input point).

    python surface_runner.py --cloud cloud.npz --out mesh.npz [--depth 9]

cloud.npz: points (N,3) f, colors (N,3) u8, labels (N,) u8 (1 observed, 2 estimated),
cameras (N,3) f (centre of the camera that saw each point), normals (N,3) f (zero = none).
mesh.npz: vertices (V,3) f32, faces (F,3) i32, colors (V,3) u8, normals (V,3) f32,
labels (V,) u8 (1 observed, 2 estimated, 3 inferred).
"""

import argparse
from pathlib import Path

import numpy as np
from _common import Timer, emit, log, run_main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--density-quantile", type=float, default=0.03)
    ap.add_argument("--max-gap", type=float, default=6.0)
    ap.add_argument("--inferred-gap", type=float, default=2.5)
    ap.add_argument("--min-component", type=float, default=0.02)
    ap.add_argument("--smooth-iterations", type=int, default=20)
    args = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    timer = Timer()
    data = np.load(args.cloud)
    pts = data["points"].astype(np.float64)
    cols = data["colors"]
    labels = data["labels"]
    cams = data["cameras"].astype(np.float64)
    prior = data["normals"].astype(np.float64)
    has_prior = np.linalg.norm(prior, axis=1) > 0.5
    if len(pts) < 1000:
        raise ValueError(f"Only {len(pts)} points: too few for a surface.")

    tree = cKDTree(pts)
    nn, _ = tree.query(pts, k=2)
    spacing = float(np.median(nn[:, 1]))

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6 * spacing, max_nn=30))
    est = np.asarray(pcd.normals)
    normals = np.where(has_prior[:, None], prior, est)
    to_cam = cams - pts
    flip = np.einsum("ij,ij->i", normals, to_cam) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    emit(event="progress", step="normals", prior_share=round(float(has_prior.mean()), 4))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth, scale=1.1, linear_fit=False
    )
    dens = np.asarray(dens)
    poisson_vertices = len(dens)
    emit(event="progress", step="poisson", vertices=poisson_vertices)

    verts = np.asarray(mesh.vertices)
    dist, _ = tree.query(verts, k=1)
    low = dens < np.quantile(dens, args.density_quantile)
    far = dist > args.max_gap * spacing
    mesh.remove_vertices_by_mask(low | far)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    small = sizes < max(50, args.min_component * sizes.max()) if len(sizes) else sizes
    mesh.remove_triangles_by_mask(small[clusters] if len(sizes) else np.zeros(0, bool))
    mesh.remove_unreferenced_vertices()
    if args.smooth_iterations > 0:
        # Per-pixel depth noise makes Poisson surfaces lumpy; Taubin smoothing flattens that
        # without shrinking the shape (unlike plain Laplacian smoothing).
        mesh = mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations)
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles, dtype=np.int32)
    if len(f) == 0:
        raise ValueError("Poisson produced no surface that the points support.")
    d, idx = tree.query(v, k=4)
    w = 1.0 / np.maximum(d, 1e-9)
    vcol = (cols[idx].astype(np.float64) * w[..., None]).sum(1) / w.sum(1)[:, None]
    vlab = np.where(d[:, 0] > args.inferred_gap * spacing, 3, labels[idx[:, 0]]).astype(np.uint8)
    np.savez(
        args.out,
        vertices=v.astype(np.float32),
        faces=f,
        colors=vcol.round().clip(0, 255).astype(np.uint8),
        normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        labels=vlab,
    )
    area = float(mesh.get_surface_area())
    emit(
        event="done",
        input_points=len(pts),
        point_spacing=round(spacing, 5),
        prior_normal_share=round(float(has_prior.mean()), 4),
        normals_flipped_to_camera=int(flip.sum()),
        poisson_depth=args.depth,
        smooth_iterations=args.smooth_iterations,
        poisson_vertices=poisson_vertices,
        removed_low_density=int(low.sum()),
        removed_unsupported=int((far & ~low).sum()),
        components_kept=int((~small).sum()) if len(sizes) else 0,
        vertices=len(v),
        triangles=len(f),
        surface_area=round(area, 3),
        watertight=bool(mesh.is_watertight()),
        vertex_classes={
            "observed": int((vlab == 1).sum()),
            "estimated": int((vlab == 2).sum()),
            "inferred": int((vlab == 3).sum()),
        },
        timings={"total_s": timer.s()},
    )
    log(f"mesh: {len(v)} vertices, {len(f)} triangles")
    return 0


if __name__ == "__main__":
    run_main(main)
