"""The height-map surface: grid, solid mesh and texture bake, on a synthetic hill whose colour
is a known function of the ground position, so the baked texture can be checked."""

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from nazar_worker.glb import read_mesh_glb, write_mesh_glb
from nazar_worker.heightfield import (
    INFERRED,
    OBSERVED,
    PAD_ROWS,
    Camera,
    HeightfieldError,
    bake_texture,
    build_grid,
    build_mesh,
    build_textured_heightfield,
)

EXTENT = 5.0  # the scene spans [-5, 5] on x and z


def height(x: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.5 * np.exp(-(x**2 + z**2) / 8.0)


def colour(x: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Smooth RGB in [0, 255]: red follows x, green follows z, blue is constant."""
    r = 40 + 170 * (x + EXTENT) / (2 * EXTENT)
    g = 40 + 170 * (z + EXTENT) / (2 * EXTENT)
    return np.stack([r, g, np.full_like(r, 120.0)], axis=-1)


def scene(
    n: int = 60_000, seed: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-EXTENT, EXTENT, n)
    z = rng.uniform(-EXTENT, EXTENT, n)
    keep = np.hypot(x - 3, z - 3) > 0.4  # a hole to be bridged
    x, z = x[keep], z[keep]
    y = height(x, z) + rng.normal(0, 0.01, len(x))
    pts = np.stack([x, y, z], axis=1)
    return pts, colour(x, z).astype(np.uint8), np.full(len(x), OBSERVED, np.uint8)


def nadir_camera(cx: float, cz: float, alt: float = 12.0, size: int = 240) -> Camera:
    """Looks straight down (OpenCV axes: z forward = -Y world, x = +X, y = +Z world)."""
    c2w = np.eye(4)
    c2w[:3, 0] = (1, 0, 0)
    c2w[:3, 1] = (0, 0, 1)
    c2w[:3, 2] = (0, -1, 0)
    c2w[:3, 3] = (cx, alt, cz)
    f = float(size)
    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    d_cam = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u)], axis=-1)
    d = d_cam @ c2w[:3, :3].T
    h = np.zeros_like(u)
    for _ in range(8):  # intersect each ray with the surface
        t = (h - alt) / d[..., 1]
        h = height(cx + t * d[..., 0], cz + t * d[..., 2])
    t = (h - alt) / d[..., 1]
    rgb = colour(cx + t * d[..., 0], cz + t * d[..., 2])
    image = np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1].copy()  # BGR
    return Camera(c2w, f, f, size / 2, size / 2, image)


def test_grid_recovers_the_surface_and_labels_the_bridged_hole() -> None:
    pts, col, lab = scene()
    # A floater far above the surface in an otherwise empty cell must not become a tower.
    pts = np.vstack([pts, [[-4.6, 20.0, -4.6], [-4.6, 20.1, -4.6]]])
    col = np.vstack([col, [[255, 0, 0]] * 2]).astype(np.uint8)
    lab = np.concatenate([lab, [OBSERVED, OBSERVED]]).astype(np.uint8)
    grid = build_grid(pts, col, lab)
    x, z = grid.node_xz()
    expected = height(x[None, :], z[:, None])
    err = np.abs(grid.height - expected)[grid.mask]
    assert np.median(err) < 0.05
    assert np.percentile(err, 99) < 0.15
    assert grid.height.max() < 3  # the floater did not survive
    assert grid.stats["spikes_removed"] >= 1
    # The hole at (3, 3) was bridged and says so.
    ix, iz = int((3 - grid.x0) / grid.cell), int((3 - grid.z0) / grid.cell)
    assert grid.mask[iz, ix]
    assert grid.labels[iz, ix] == INFERRED
    assert grid.labels[grid.mask].tolist().count(INFERRED) == grid.stats["bridged_cells"]


def test_too_few_points_or_cells_is_refused_with_a_reason() -> None:
    pts, col, lab = scene(n=300)
    with pytest.raises(HeightfieldError, match="too few"):
        build_grid(pts, col, lab)
    pts, col, lab = scene(n=3000)
    with pytest.raises(HeightfieldError):
        build_grid(pts, col, lab, min_cells=10_000)


def test_the_mesh_is_a_closed_solid_with_valid_normals_and_uvs() -> None:
    pts, col, lab = scene()
    grid = build_grid(pts, col, lab)
    mesh = build_mesh(grid, texture=None)
    assert mesh.closed
    assert mesh.faces.max() < len(mesh.vertices)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-4)
    assert mesh.uv.min() >= 0 and mesh.uv.max() <= 1
    # Top faces point up, the base faces down.
    assert (mesh.normals[: mesh.stats["surface_vertices"], 1] > 0).all()
    assert mesh.vertices[:, 1].min() < grid.height[grid.mask].min()  # a base below the surface
    assert mesh.stats["inferred_surface_share"] > 0  # the bridged hole is counted
    assert mesh.texture is None


def test_texture_is_baked_from_the_frames_at_the_right_place() -> None:
    pts, col, lab = scene()
    grid = build_grid(pts, col, lab)
    cams = [nadir_camera(-2, 0), nadir_camera(2, 0)]
    tex, stats = bake_texture(grid, cams)
    assert tex.shape[0] == stats["size"][1] + PAD_ROWS
    assert stats["coverage"] > 0.9
    th, tw = stats["size"][1], stats["size"][0]
    rng = np.random.default_rng(3)
    rows = rng.integers(10, th - 10, 400)
    cols = rng.integers(10, tw - 10, 400)
    xw = grid.x0 + (cols + 0.5) * grid.cell * grid.shape[1] / tw
    zw = grid.z0 + (rows + 0.5) * grid.cell * grid.shape[0] / th
    inside = (np.abs(xw) < EXTENT - 1) & (np.abs(zw) < EXTENT - 1)
    want = colour(xw[inside], zw[inside])
    got = tex[rows[inside], cols[inside]].astype(np.float64)  # the texture is RGB
    assert np.abs(got - want).mean() < 12
    # The neutral strip used by the solid's base is at the bottom.
    assert (tex[-PAD_ROWS:] == tex[-1, 0]).all()


def test_a_tall_pillar_hides_the_ground_behind_it_from_a_low_camera() -> None:
    """Texels behind a pillar, as seen by the only camera, must not receive the pillar's
    colour: occluded points are left out, not painted with whatever is in front."""
    pts, col, lab = scene()
    ring = np.random.default_rng(5).uniform(-0.3, 0.3, (4000, 2))
    pillar = np.column_stack(
        [ring[:, 0], np.random.default_rng(6).uniform(1.5, 6.0, 4000), ring[:, 1] - 2.0]
    )
    pts = np.vstack([pts, pillar])
    col = np.vstack([col, np.tile([255, 255, 255], (4000, 1))]).astype(np.uint8)
    lab = np.concatenate([lab, np.full(4000, OBSERVED)]).astype(np.uint8)
    grid = build_grid(pts, col, lab)
    assert grid.height.max() > 4  # the pillar is in the height map

    # A camera low and to the +z side of the pillar looking toward -z and down a little.
    pos = np.array([0.0, 4.0, 4.0])
    fwd = np.array([0.0, -0.45, -1.0])
    fwd /= np.linalg.norm(fwd)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, pos
    img = np.full((240, 240, 3), (0, 255, 0), np.uint8)  # everything green (BGR)
    cam = Camera(c2w, 240.0, 240.0, 120.0, 120.0, img)
    _, stats = bake_texture(grid, [cam])
    assert stats["coverage"] < 0.9  # not everything is seen


def test_a_camera_that_sees_nothing_is_a_texture_error_not_a_crash() -> None:
    pts, col, lab = scene()
    away = nadir_camera(0, 0, alt=-12.0)  # below the scene, looking down: sees nothing
    mesh, notes = build_textured_heightfield(pts, col, lab, [away])
    assert mesh.texture is None
    assert "texture_error" in notes
    assert mesh.closed
    none_mesh, none_notes = build_textured_heightfield(pts, col, lab, None)
    assert none_mesh.texture is None and "texture_error" in none_notes


def test_end_to_end_textured_glb_round_trips() -> None:
    pts, col, lab = scene()
    mesh, notes = build_textured_heightfield(
        pts, col, lab, [nadir_camera(-2, 0), nadir_camera(2, 0)]
    )
    assert mesh.texture is not None and "texture_error" not in notes
    ok, png = cv2.imencode(".png", mesh.texture[..., ::-1])
    assert ok
    data = write_mesh_glb(
        mesh.vertices, mesh.faces, mesh.colors, mesh.normals, uv=mesh.uv, texture=png.tobytes()
    )
    back = read_mesh_glb(data)
    assert back.colors is None  # a textured mesh has no vertex colours to multiply the photo
    assert back.uv is not None and back.uv.shape == (len(mesh.vertices), 2)
    assert (back.vertices == mesh.vertices).all() and (back.faces == mesh.faces).all()
    assert back.texture is not None
    decoded = cv2.imdecode(np.frombuffer(back.texture, np.uint8), cv2.IMREAD_COLOR)
    assert mesh.texture is not None and decoded is not None
    assert decoded.shape == mesh.texture.shape
    assert back.doc["extensionsUsed"] == ["KHR_materials_unlit"]
    # And without a texture the vertex colours are written as before.
    plain = read_mesh_glb(write_mesh_glb(mesh.vertices, mesh.faces, mesh.colors, mesh.normals))
    assert plain.colors is not None and plain.uv is None and plain.texture is None
