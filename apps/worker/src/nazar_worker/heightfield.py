"""A textured, solid surface from a single pass of aerial video.

Why a height map. One forward drone pass sees the ground and the tops of things from one side:
the scene is a surface z = h(x, y), not a free-standing object. Poisson meshing of a noisy cloud
of per-pixel guesses (the previous approach) turns that noise into lumps and invents surface
where nothing was seen. Fusing every point into a regular grid and keeping the upper envelope
of each cell gives a smooth, consistent surface by construction, and a regular grid is also a
natural place to hang a photographic texture.

The steps (all CPU, numpy/scipy/OpenCV; no extra model or download):

1. `build_grid`: the fused points (+Y up) are binned on the ground plane; each cell keeps a high
   percentile of its heights (the visible top, robust to floaters), isolated spikes are
   removed, small holes are filled by normalised convolution (and labelled *inferred*), and
   the heights are smoothed with an edge-preserving filter so real walls survive.
2. `bake_texture`: every texel of a top-down texture is a 3D point on the surface; it is
   projected into each video frame (full resolution), checked for occlusion against a depth
   buffer of the surface itself, and coloured from the frames that see it best (steeper
   viewing angle, closer, away from the image border), blended softly so seams are hidden
   without averaging misregistered views into blur.
3. `build_mesh`: the grid becomes triangles, closed into a solid block (side walls and a flat
   base) so it reads as a physical model, with planar UVs into the texture.

What this does not do: it cannot show what no frame saw. Vertical faces become steep slopes,
and the unseen far side is simply not there. Cells bridged from neighbours are labelled
inferred, never presented as observed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage

OBSERVED, ESTIMATED, INFERRED = 1, 2, 3
"""Per-vertex confidence class, the same codes as the rest of the pipeline."""

BASE_RGB = (58, 58, 58)
PAD_ROWS = 8
"""Rows appended to the texture and filled with BASE_RGB: the colour of the solid's base."""


class HeightfieldError(Exception):
    """The scene cannot be turned into a height-map surface (the reason is the message)."""


@dataclass(frozen=True)
class Camera:
    """One video frame: pose (OpenCV axes: x right, y down, z forward), intrinsics in pixels
    of the full-resolution `image` (BGR, uint8)."""

    c2w: npt.NDArray[np.float64]
    fx: float
    fy: float
    cx: float
    cy: float
    image: npt.NDArray[np.uint8]


@dataclass
class Grid:
    x0: float
    z0: float
    cell: float
    height: npt.NDArray[np.float32]
    """(nz, nx) heights; meaningful inside `mask`, nearest-filled elsewhere."""
    mask: npt.NDArray[np.bool_]
    """Cells that become surface (observed or bridged)."""
    labels: npt.NDArray[np.uint8]
    """(nz, nx): OBSERVED / ESTIMATED from the points, INFERRED for bridged cells."""
    colors: npt.NDArray[np.uint8]
    """(nz, nx, 3) RGB, the fallback when no texture can be made."""
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.height.shape[0]), int(self.height.shape[1])

    def node_xz(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        nz, nx = self.shape
        x = self.x0 + (np.arange(nx, dtype=np.float64) + 0.5) * self.cell
        z = self.z0 + (np.arange(nz, dtype=np.float64) + 0.5) * self.cell
        return np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)


@dataclass
class HeightfieldMesh:
    vertices: npt.NDArray[np.float32]
    faces: npt.NDArray[np.uint32]
    uv: npt.NDArray[np.float32]
    normals: npt.NDArray[np.float32]
    colors: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.uint8]
    texture: npt.NDArray[np.uint8] | None
    """(rows, cols, 3) RGB, or None when no texture could be baked."""
    closed: bool
    stats: dict[str, Any]


# ---------------------------------------------------------------------------------------
# 1. The grid


def _nearest_fill(values: npt.NDArray[Any], valid: npt.NDArray[np.bool_]) -> npt.NDArray[Any]:
    """`values` with every invalid cell replaced by the nearest valid one."""
    if valid.all():
        return values
    _, (iz, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return np.asarray(values[iz, ix])


def _isolated_spikes(
    height: npt.NDArray[np.float32], valid: npt.NDArray[np.bool_], thr: float
) -> npt.NDArray[np.bool_]:
    """Valid cells that no more than one valid neighbour agrees with (within `thr`): floaters
    and one-cell towers, not walls (a wall has neighbours at its own height)."""
    support = np.zeros(height.shape, dtype=np.int32)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dz == 0 and dx == 0:
                continue
            h = np.roll(height, (dz, dx), axis=(0, 1))
            v = np.roll(valid, (dz, dx), axis=(0, 1))
            support += (v & (np.abs(h - height) <= thr)).astype(np.int32)
    return valid & (support <= 1)


def build_grid(
    points: npt.NDArray[np.floating[Any]],
    colors: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    *,
    percentile: float = 0.9,
    min_points: int = 2,
    fill_radius: int = 6,
    max_cells: int = 600,
    target_points: float = 8.0,
    min_cells: int = 400,
    min_coverage: float = 0.25,
) -> Grid:
    """Bin points (+Y up) on the ground plane into a height map. See the module docstring."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 1000:
        raise HeightfieldError(f"Only {len(pts)} points: too few for a surface.")
    xz = pts[:, [0, 2]]
    lo = np.percentile(xz, 0.5, axis=0)
    hi = np.percentile(xz, 99.5, axis=0)
    span = hi - lo
    extent = float(span.max())
    if extent <= 0:
        raise HeightfieldError("The points have no horizontal extent.")

    # Cell size from the point density: about `target_points` points per occupied cell, so the
    # per-cell height is a robust statistic rather than a single noisy sample.
    coarse = 96
    ci = np.clip(((pts[:, 0] - lo[0]) / span[0] * (coarse - 1)).astype(np.int64), 0, coarse - 1)
    cj = np.clip(((pts[:, 2] - lo[1]) / span[1] * (coarse - 1)).astype(np.int64), 0, coarse - 1)
    occupied = np.unique(cj * coarse + ci).size / (coarse * coarse)
    area = max(occupied * span[0] * span[1], 1e-12)
    cell = max(float(np.sqrt(target_points * area / len(pts))), extent / max_cells)
    nx = int(np.ceil(span[0] / cell)) + 1
    nz = int(np.ceil(span[1] / cell)) + 1

    ix = np.floor((pts[:, 0] - lo[0]) / cell).astype(np.int64)
    iz = np.floor((pts[:, 2] - lo[1]) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    p, col, lab = pts[inside], colors[inside], labels[inside]
    cid = iz[inside] * nx + ix[inside]
    order = np.lexsort((p[:, 1], cid))
    cid, y, col, lab = cid[order], p[order, 1], col[order].astype(np.float64), lab[order]
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    counts = np.diff(np.r_[starts, len(cid)])
    pick = starts + np.minimum(counts - 1, np.floor(percentile * (counts - 1)).astype(np.int64))
    cells = cid[starts]

    height_f = np.full(nz * nx, np.nan, dtype=np.float32)
    height_f[cells] = y[pick]
    # Colour: mean of the upper half of each cell's points (the visible side).
    rank = np.arange(len(cid)) - np.repeat(starts, counts)
    upper = (rank >= np.repeat((counts - 1) // 2, counts)).astype(np.float64)
    csum = np.add.reduceat(col * upper[:, None], starts, axis=0)
    cmean = csum / np.maximum(np.add.reduceat(upper, starts), 1.0)[:, None]
    rgb_f = np.zeros((nz * nx, 3), dtype=np.float64)
    rgb_f[cells] = cmean
    lab_f = np.zeros(nz * nx, dtype=np.uint8)
    lab_f[cells] = lab[pick]
    valid_f = np.zeros(nz * nx, dtype=bool)
    valid_f[cells] = counts >= min_points

    height: npt.NDArray[np.float32] = height_f.reshape(nz, nx)
    valid: npt.NDArray[np.bool_] = valid_f.reshape(nz, nx)
    cell_rgb: npt.NDArray[np.float64] = rgb_f.reshape(nz, nx, 3)
    cell_lab: npt.NDArray[np.uint8] = lab_f.reshape(nz, nx)
    if int(valid.sum()) < min_cells:
        raise HeightfieldError(
            f"Only {int(valid.sum())} ground cells have points: too sparse for a height map."
        )
    height = np.where(valid, height, 0.0).astype(np.float32)
    h_lo, h_hi = (float(v) for v in np.percentile(height[valid], [1, 99]))
    y_range = max(h_hi - h_lo, 1e-6)

    # Isolated spikes (floaters) are dropped before anything is filled from them.
    spikes = _isolated_spikes(height, valid, thr=max(4.0 * cell, 0.03 * y_range))
    valid &= ~spikes
    observed_cells = int(valid.sum())
    if observed_cells < min_cells:
        raise HeightfieldError("Too few cells are left after removing isolated points.")

    # Bridge small holes by normalised convolution; larger gaps stay empty (not invented).
    dist = ndimage.distance_transform_edt(~valid)
    fillable = ~valid & (dist <= fill_radius)
    w = valid.astype(np.float64)
    sigma = max(1.0, fill_radius / 2.0)
    num = ndimage.gaussian_filter(np.where(valid, height, 0.0).astype(np.float64) * w, sigma)
    den = ndimage.gaussian_filter(w, sigma)
    bridged = np.where(den > 1e-4, num / np.maximum(den, 1e-12), 0.0).astype(np.float32)
    fillable &= den > 1e-4
    mask = valid | fillable
    # Keep only the main body: drop islands under 1 % of the largest piece.
    parts, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    areas = np.bincount(comp.ravel().astype(np.int64), minlength=parts)
    areas[0] = 0
    if areas.max() == 0:
        raise HeightfieldError("No connected surface was found.")
    keep = areas >= max(50, 0.01 * areas.max())
    keep[0] = False
    mask &= keep[comp]
    valid &= mask
    fillable &= mask
    coverage = observed_cells / max(int(mask.sum()), 1)
    if coverage < min_coverage:
        raise HeightfieldError(
            f"Only {coverage:.0%} of the surface has points under it: it would be mostly invented."
        )

    h: npt.NDArray[np.float32] = np.where(valid, height, bridged).astype(np.float32)
    h = _nearest_fill(h, mask)
    # Edge-preserving smoothing: noise is flattened, real steps (walls) larger than the
    # tolerance are kept.
    sigma_color = max(0.03 * y_range, 2.0 * cell)
    for _ in range(2):
        h = np.asarray(cv2.medianBlur(h, 3), dtype=np.float32)
        h = np.asarray(cv2.bilateralFilter(h, 5, sigma_color, 1.5), dtype=np.float32)

    rgb = np.clip(cell_rgb, 0, 255).round().astype(np.uint8)
    rgb = _nearest_fill(rgb, valid)
    labels_out = np.where(valid, cell_lab, INFERRED).astype(np.uint8)
    labels_out[~mask] = 0
    return Grid(
        x0=float(lo[0]),
        z0=float(lo[1]),
        cell=float(cell),
        height=h,
        mask=mask,
        labels=labels_out,
        colors=rgb,
        stats={
            "cells": int(mask.sum()),
            "observed_cells": int(valid.sum()),
            "bridged_cells": int((mask & ~valid).sum()),
            "spikes_removed": int(spikes.sum()),
            "cell_size": round(float(cell), 6),
            "points_per_cell": round(float(len(pts) / max(int(mask.sum()), 1)), 2),
            "grid": [int(nz), int(nx)],
            "height_range": [round(h_lo, 4), round(h_hi, 4)],
            "coverage": round(float(coverage), 4),
        },
    )


# ---------------------------------------------------------------------------------------
# 2. The texture


def _bilinear(
    image: npt.NDArray[np.uint8], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.float32]:
    """Bilinear colour of `image` (H, W, 3) at pixel coordinates (u, v), all inside the image
    by at least one pixel. Pixel centres are at integer coordinates + 0.5."""
    x = u - 0.5
    y = v - 0.5
    x0 = np.clip(np.floor(x).astype(np.int64), 0, image.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, image.shape[0] - 2)
    fx = np.clip(x - x0, 0.0, 1.0).astype(np.float32)[:, None]
    fy = np.clip(y - y0, 0.0, 1.0).astype(np.float32)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return np.asarray(top * (1 - fy) + bottom * fy, dtype=np.float32)


def _zbuffer(
    z: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    width: int,
    height: int,
    block: int,
) -> npt.NDArray[np.float32]:
    """Nearest surface depth per `block`-pixel cell of one camera, from the surface nodes."""
    bh, bw = -(-height // block), -(-width // block)
    buf = np.full((bh, bw), np.inf, dtype=np.float32)
    ok = (z > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    np.minimum.at(buf, ((v[ok] / block).astype(np.int64), (u[ok] / block).astype(np.int64)), z[ok])
    # A one-cell minimum filter closes pinholes between neighbouring nodes.
    big = np.where(np.isinf(buf), np.float32(1e30), buf).astype(np.float32)
    return np.asarray(cv2.erode(big, np.ones((3, 3), np.uint8)))


def bake_texture(
    grid: Grid,
    cameras: list[Camera],
    *,
    max_side: int = 3072,
    blend_power: float = 6.0,
    min_cos: float = 0.12,
    band: int = 192,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[npt.NDArray[np.uint8], dict[str, Any]]:
    """Top-down texture of the surface from the full-resolution frames.

    The texel size follows the frames' own ground resolution (metres per pixel at the surface),
    not the coarse grid: a 1080p frame gives a texture several times finer than the mesh.
    Returns (texture RGB with `PAD_ROWS` neutral rows appended, stats). `on_progress(done,
    total)` is called after each band of rows (it may raise to cancel).
    """
    if not cameras:
        raise HeightfieldError("No camera is available to texture the surface.")
    nz, nx = grid.shape

    # Per-camera constants and a depth buffer of the surface nodes.
    node_x, node_z = grid.node_xz()
    ii, jj = np.nonzero(grid.mask)
    nodes = np.stack([node_x[jj], grid.height[ii, jj].astype(np.float64), node_z[ii]], axis=1)
    prepared = []
    gsds = []
    for cam in cameras:
        h_img, w_img = cam.image.shape[:2]
        rot, pos = cam.c2w[:3, :3], cam.c2w[:3, 3]
        pc = (nodes - pos) @ rot
        zz = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cam.fx * pc[:, 0] / zz + cam.cx
            v = cam.fy * pc[:, 1] / zz + cam.cy
        in_view = (zz > 0.05) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        if in_view.any():
            gsds.append(float(np.median(zz[in_view])) / (0.5 * (cam.fx + cam.fy)))
        block = max(2, round(min(w_img, h_img) / 240))
        prepared.append((cam, rot, pos, _zbuffer(zz, u, v, w_img, h_img, block), block))
    if not gsds:
        raise HeightfieldError("No video frame sees the surface (check the camera poses).")

    extent = max(nx, nz) * grid.cell
    texel_size = float(np.clip(np.median(gsds), extent / max_side, grid.cell / 2.0))
    tw, th = max(8, round(nx * grid.cell / texel_size)), max(8, round(nz * grid.cell / texel_size))
    mask_t = cv2.resize(grid.mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    height_t = cv2.resize(grid.height, (tw, th), interpolation=cv2.INTER_LINEAR)
    texel = grid.cell * nx / tw, grid.cell * nz / th  # world size of a texel in x and z
    gz, gx = (np.asarray(g) for g in np.gradient(height_t.astype(np.float64), texel[1], texel[0]))
    xs = grid.x0 + (np.arange(tw) + 0.5) * texel[0]

    acc = np.zeros((th, tw, 3), dtype=np.float32)
    wsum = np.zeros((th, tw), dtype=np.float32)
    for r0 in range(0, th, band):
        r1 = min(th, r0 + band)
        zr = grid.z0 + (np.arange(r0, r1) + 0.5) * texel[1]
        grid_p = np.empty((r1 - r0, tw, 3), dtype=np.float64)
        grid_p[..., 0] = xs[None, :]
        grid_p[..., 1] = height_t[r0:r1]
        grid_p[..., 2] = zr[:, None]
        p = grid_p.reshape(-1, 3)
        normal = np.stack([-gx[r0:r1].ravel(), np.ones(p.shape[0]), -gz[r0:r1].ravel()], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        live = mask_t[r0:r1].ravel()
        band_acc = np.zeros((p.shape[0], 3), dtype=np.float32)
        band_w = np.zeros(p.shape[0], dtype=np.float32)
        for cam, rot, pos, zbuf, block in prepared:
            h_img, w_img = cam.image.shape[:2]
            to_cam = pos - p
            dist = np.linalg.norm(to_cam, axis=1)
            cos = np.einsum("ij,ij->i", normal, to_cam) / np.maximum(dist, 1e-9)
            pc = (p - pos) @ rot
            zz = pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = cam.fx * pc[:, 0] / zz + cam.cx
                v = cam.fy * pc[:, 1] / zz + cam.cy
            ok = live & (zz > 0.05) & (cos > min_cos)
            ok &= (u >= 1) & (u <= w_img - 2) & (v >= 1) & (v <= h_img - 2)
            if not ok.any():
                continue
            sel = np.flatnonzero(ok)
            zb = zbuf[(v[sel] / block).astype(np.int64), (u[sel] / block).astype(np.int64)]
            tol = 0.03 * zz[sel] + 2.0 * block * zz[sel] / (0.5 * (cam.fx + cam.fy))
            seen = zz[sel] <= zb + tol
            sel = sel[seen]
            if sel.size == 0:
                continue
            border = np.minimum.reduce([u[sel], w_img - 1 - u[sel], v[sel], h_img - 1 - v[sel]]) / (
                0.06 * min(w_img, h_img)
            )
            weight = (cos[sel] ** 2 / zz[sel]) * np.clip(border, 0.0, 1.0)
            keep = weight > 0
            sel, weight = sel[keep], weight[keep]
            if sel.size == 0:
                continue
            sampled = _bilinear(cam.image, u[sel], v[sel])
            rgb = sampled[:, ::-1]  # BGR -> RGB
            wp = (weight**blend_power).astype(np.float32)
            band_acc[sel] += rgb * wp[:, None]
            band_w[sel] += wp
        acc[r0:r1] = band_acc.reshape(r1 - r0, tw, 3)
        wsum[r0:r1] = band_w.reshape(r1 - r0, tw)
        if on_progress is not None:
            on_progress(r1, th)

    seen_t = wsum > 0
    if not (seen_t & mask_t).any():
        raise HeightfieldError("No video frame sees the surface (check the camera poses).")
    tex = np.where(seen_t[..., None], acc / np.maximum(wsum, 1e-30)[..., None], 0.0)
    tex = np.clip(tex, 0, 255).round().astype(np.uint8)
    coverage = float((seen_t & mask_t).sum() / max(int(mask_t.sum()), 1))
    tex = _nearest_fill(tex, seen_t)  # unseen texels and the margin take the nearest colour
    pad = np.empty((PAD_ROWS, tw, 3), dtype=np.uint8)
    pad[:] = BASE_RGB
    return np.concatenate([tex, pad], axis=0), {
        "size": [int(tw), int(th)],
        "texel_size": round(float(texel[0]), 6),
        "source_pixel_size": round(float(np.median(gsds)), 6),
        "views": len(cameras),
        "coverage": round(coverage, 4),
    }


# ---------------------------------------------------------------------------------------
# 3. The solid mesh


def _closed(faces: npt.NDArray[np.integer[Any]], weld: npt.NDArray[np.int64]) -> bool:
    """Every edge is shared by exactly two faces, once vertices at the same place (UV seams)
    are welded."""
    f = weld[faces.astype(np.int64)]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    key = np.minimum(e[:, 0], e[:, 1]) * (int(weld.max()) + 1) + np.maximum(e[:, 0], e[:, 1])
    _, counts = np.unique(key, return_counts=True)
    return bool((counts == 2).all())


def build_mesh(
    grid: Grid, texture: npt.NDArray[np.uint8] | None, base_margin: float = 0.08
) -> HeightfieldMesh:
    """Triangles for the surface, closed into a solid block, with UVs into `texture`."""
    nz, nx = grid.shape
    mask = grid.mask
    ids = np.full(mask.shape, -1, dtype=np.int64)
    ii, jj = np.nonzero(mask)
    n_top = len(ii)
    ids[ii, jj] = np.arange(n_top)
    node_x, node_z = grid.node_xz()
    h = grid.height.astype(np.float64)
    top = np.stack([node_x[jj], h[ii, jj], node_z[ii]], axis=1)

    quad = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
    qi, qj = np.nonzero(quad)
    if len(qi) == 0:
        raise HeightfieldError("The surface has no connected cells to triangulate.")
    v00, v01 = ids[qi, qj], ids[qi, qj + 1]
    v10, v11 = ids[qi + 1, qj], ids[qi + 1, qj + 1]
    diag_a = np.abs(h[qi, qj] - h[qi + 1, qj + 1]) <= np.abs(h[qi, qj + 1] - h[qi + 1, qj])
    t1 = np.where(diag_a[:, None], np.stack([v00, v10, v11], 1), np.stack([v00, v10, v01], 1))
    t2 = np.where(diag_a[:, None], np.stack([v00, v11, v01], 1), np.stack([v01, v10, v11], 1))
    top_faces = np.concatenate([t1, t2], axis=0)
    used = np.zeros(n_top, dtype=bool)
    used[top_faces.ravel()] = True
    if not used.all():  # nodes with no complete quad around them are dropped
        remap = np.cumsum(used) - 1
        top, ii, jj = top[used], ii[used], jj[used]
        top_faces = remap[top_faces]
        n_top = len(top)

    # Smooth upward normals from the height gradient.
    gz, gx = (np.asarray(g) for g in np.gradient(h, grid.cell))
    n_top_vec = np.stack([-gx[ii, jj], np.ones(n_top), -gz[ii, jj]], axis=1)
    n_top_vec /= np.linalg.norm(n_top_vec, axis=1, keepdims=True)

    # Boundary edges of the top surface get a wall down to the base.
    e = np.concatenate([top_faces[:, [0, 1]], top_faces[:, [1, 2]], top_faces[:, [2, 0]]], axis=0)
    key = np.minimum(e[:, 0], e[:, 1]) * n_top + np.maximum(e[:, 0], e[:, 1])
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    boundary = e[counts[inverse] == 1]
    wall_verts = np.unique(boundary.ravel())
    n_wall = len(wall_verts)
    base_y = float(top[:, 1].min() - base_margin * max(nx, nz) * grid.cell)
    wall_index = n_top + np.searchsorted(wall_verts, boundary)  # (B, 2) wall-bottom ids
    a, b = boundary[:, 0], boundary[:, 1]
    a_w, b_w = wall_index[:, 0], wall_index[:, 1]
    wall_faces = np.concatenate([np.stack([b, a, a_w], 1), np.stack([b, a_w, b_w], 1)], axis=0)
    base_offset = n_top + n_wall
    base_faces = top_faces[:, [0, 2, 1]] + base_offset

    wall_pos = top[wall_verts].copy()
    wall_pos[:, 1] = base_y
    base_pos = top.copy()
    base_pos[:, 1] = base_y
    vertices = np.concatenate([top, wall_pos, base_pos], axis=0).astype(np.float32)
    faces = np.concatenate([top_faces, wall_faces, base_faces], axis=0).astype(np.uint32)

    # Wall normals: horizontal, outward (interior lies to the left of a boundary edge).
    d = top[b] - top[a]
    out = np.stack([-d[:, 2], np.zeros(len(d)), d[:, 0]], axis=1)
    wall_normal = np.zeros((n_wall, 3))
    np.add.at(wall_normal, np.searchsorted(wall_verts, a), out)
    np.add.at(wall_normal, np.searchsorted(wall_verts, b), out)
    norm = np.linalg.norm(wall_normal, axis=1, keepdims=True)
    wall_normal = wall_normal / np.maximum(norm, 1e-12)
    base_normal = np.tile([0.0, -1.0, 0.0], (n_top, 1))
    normals = np.concatenate([n_top_vec, wall_normal, base_normal], axis=0).astype(np.float32)

    # Planar UVs from above; the base points into the texture's neutral strip.
    rows = texture.shape[0] if texture is not None else nz
    tex_rows = rows - PAD_ROWS if texture is not None else rows
    pad = PAD_ROWS if texture is not None else 0
    u = (jj + 0.5) / nx
    vv = (ii + 0.5) / nz * tex_rows / (tex_rows + pad)
    uv_top = np.stack([u, vv], axis=1)
    uv_wall = uv_top[wall_verts]
    base_v = (tex_rows + pad / 2) / (tex_rows + pad)
    uv_base = np.tile([0.5, base_v], (n_top, 1))
    uv = np.concatenate([uv_top, uv_wall, uv_base], axis=0).astype(np.float32)

    top_rgb = grid.colors[ii, jj]
    colors = np.concatenate(
        [top_rgb, top_rgb[wall_verts], np.tile(np.array(BASE_RGB, np.uint8), (n_top, 1))], axis=0
    ).astype(np.uint8)
    top_lab = grid.labels[ii, jj]
    labels = np.concatenate([top_lab, top_lab[wall_verts], np.full(n_top, INFERRED, np.uint8)])

    weld = np.arange(len(vertices), dtype=np.int64)
    weld[base_offset + wall_verts] = n_top + np.arange(n_wall)  # base ring == wall bottoms
    closed = _closed(faces, weld)
    inferred = float((top_lab == INFERRED).mean())
    return HeightfieldMesh(
        vertices=vertices,
        faces=faces,
        uv=uv,
        normals=normals,
        colors=colors,
        labels=labels.astype(np.uint8),
        texture=texture,
        closed=closed,
        stats={
            "vertices": len(vertices),
            "triangles": len(faces),
            "surface_vertices": int(n_top),
            "inferred_surface_share": round(inferred, 4),
            "base_y": round(base_y, 5),
        },
    )


def build_textured_heightfield(
    points: npt.NDArray[np.floating[Any]],
    colors: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    cameras: list[Camera] | None,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    **grid_options: Any,
) -> tuple[HeightfieldMesh, dict[str, Any]]:
    """Grid -> texture -> solid mesh. A texture that cannot be made (no cameras, or no frame
    sees the surface) is reported in the returned notes and the mesh falls back to vertex
    colours; a surface that cannot be made raises `HeightfieldError`."""
    grid = build_grid(points, colors, labels, **grid_options)
    notes: dict[str, Any] = {"grid": grid.stats}
    texture = None
    if cameras:
        try:
            texture, notes["texture"] = bake_texture(grid, cameras, on_progress=on_progress)
        except HeightfieldError as exc:
            notes["texture_error"] = str(exc)
    else:
        notes["texture_error"] = "No camera poses were available to texture the surface."
    mesh = build_mesh(grid, texture)
    notes["mesh"] = mesh.stats
    return mesh, notes
