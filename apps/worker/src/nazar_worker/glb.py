"""Minimal glTF 2.0 binary (GLB) writer/reader for the final point cloud and camera path.

Written with numpy from the Khronos glTF 2.0 specification, so the worker needs no extra
dependency. The file holds one scene with two nodes:

- `point_cloud`: a POINTS primitive with POSITION (float32) and COLOR_0 (normalised uint8 RGBA,
  sRGB as stored by the PLY);
- `camera_path`: a LINE_STRIP through the camera centres (omitted with fewer than two).

glTF is right-handed with +Y up, the same axes as NAZAR's local frame, so coordinates are
written unchanged. glTF viewers assume metres: when the scale is unknown the file says so in
`asset.extras`, and coordinates are unitless.
"""

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

MAGIC = 0x46546C67  # "glTF"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
POINTS, LINE_STRIP = 0, 3
FLOAT, UNSIGNED_BYTE = 5126, 5121
ARRAY_BUFFER = 34962


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


def write_glb(
    xyz: npt.NDArray[np.float64],
    rgb: npt.NDArray[np.uint8],
    path: npt.NDArray[np.float64] | None = None,
    *,
    extras: dict[str, Any] | None = None,
) -> bytes:
    pts = np.ascontiguousarray(xyz, dtype=np.float32)
    if len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError("GLB export needs at least one point and finite coordinates.")
    rgba = np.empty((len(pts), 4), dtype=np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = 255
    blobs: list[bytes] = [pts.tobytes(), rgba.tobytes()]
    accessors: list[dict[str, Any]] = [
        {
            "bufferView": 0,
            "componentType": FLOAT,
            "count": len(pts),
            "type": "VEC3",
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
        },
        {
            "bufferView": 1,
            "componentType": UNSIGNED_BYTE,
            "normalized": True,
            "count": len(pts),
            "type": "VEC4",
        },
    ]
    meshes: list[dict[str, Any]] = [
        {
            "name": "point_cloud",
            "primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": POINTS}],
        }
    ]
    nodes: list[dict[str, Any]] = [{"name": "point_cloud", "mesh": 0}]
    if path is not None and len(path) >= 2:
        line = np.ascontiguousarray(path, dtype=np.float32)
        blobs.append(line.tobytes())
        accessors.append(
            {
                "bufferView": 2,
                "componentType": FLOAT,
                "count": len(line),
                "type": "VEC3",
                "min": line.min(axis=0).tolist(),
                "max": line.max(axis=0).tolist(),
            }
        )
        meshes.append(
            {
                "name": "camera_path",
                "primitives": [{"attributes": {"POSITION": 2}, "mode": LINE_STRIP}],
            }
        )
        nodes.append({"name": "camera_path", "mesh": 1})

    views, offset, body = [], 0, b""
    for blob in blobs:
        views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": ARRAY_BUFFER}
        )
        padded = _pad(blob, b"\0")
        body += padded
        offset += len(padded)
    doc = {
        "asset": {"version": "2.0", "generator": "NAZAR worker", "extras": extras or {}},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(body)}],
    }
    js = _pad(json.dumps(doc, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(body)
    return b"".join(
        [
            struct.pack("<III", MAGIC, 2, total),
            struct.pack("<II", len(js), JSON_CHUNK),
            js,
            struct.pack("<II", len(body), BIN_CHUNK),
            body,
        ]
    )


TRIANGLES = 4
UNSIGNED_INT = 5125
ELEMENT_ARRAY_BUFFER = 34963


def write_mesh_glb(
    vertices: npt.NDArray[np.floating[Any]],
    faces: npt.NDArray[np.integer[Any]],
    colors: npt.NDArray[np.uint8],
    normals: npt.NDArray[np.floating[Any]] | None = None,
    *,
    uv: npt.NDArray[np.floating[Any]] | None = None,
    texture: bytes | None = None,
    texture_mime: str = "image/png",
    extras: dict[str, Any] | None = None,
) -> bytes:
    """One TRIANGLES mesh with POSITION, NORMAL (optional) and either a baseColor texture
    (`uv` + encoded `texture` image, shown unlit so the photograph is not re-shaded) or COLOR_0
    (normalised uint8 RGBA), indexed by uint32. Same axes and unit conventions as
    `write_glb`. With a texture there is no COLOR_0: viewers multiply the two."""
    pos = np.ascontiguousarray(vertices, dtype=np.float32)
    idx = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)
    if len(pos) == 0 or len(idx) == 0 or not np.isfinite(pos).all():
        raise ValueError("Mesh GLB export needs vertices, faces and finite coordinates.")
    if idx.max() >= len(pos):
        raise ValueError("Face index out of range.")
    textured = texture is not None
    if textured and (uv is None or len(uv) != len(pos)):
        raise ValueError("A textured mesh needs one UV per vertex.")

    blobs: list[tuple[bytes, int | None]] = []
    accessors: list[dict[str, Any]] = []

    def add(blob: bytes, target: int | None, accessor: dict[str, Any] | None) -> int:
        blobs.append((blob, target))
        if accessor is not None:
            accessors.append({"bufferView": len(blobs) - 1, **accessor})
            return len(accessors) - 1
        return len(blobs) - 1

    indices = add(
        idx.tobytes(),
        ELEMENT_ARRAY_BUFFER,
        {"componentType": UNSIGNED_INT, "count": len(idx), "type": "SCALAR"},
    )
    position = add(
        pos.tobytes(),
        ARRAY_BUFFER,
        {
            "componentType": FLOAT,
            "count": len(pos),
            "type": "VEC3",
            "min": pos.min(axis=0).tolist(),
            "max": pos.max(axis=0).tolist(),
        },
    )
    attributes = {"POSITION": position}
    if normals is not None:
        nrm = np.ascontiguousarray(normals, dtype=np.float32)
        attributes["NORMAL"] = add(
            nrm.tobytes(), ARRAY_BUFFER, {"componentType": FLOAT, "count": len(nrm), "type": "VEC3"}
        )
    primitive: dict[str, Any] = {"attributes": attributes, "indices": indices, "mode": TRIANGLES}
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "NAZAR worker", "extras": extras or {}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "surface", "mesh": 0}],
    }
    if textured:
        assert uv is not None and texture is not None
        tex_uv = np.ascontiguousarray(uv, dtype=np.float32)
        attributes["TEXCOORD_0"] = add(
            tex_uv.tobytes(),
            ARRAY_BUFFER,
            {"componentType": FLOAT, "count": len(tex_uv), "type": "VEC2"},
        )
        image_view = add(texture, None, None)
        primitive["material"] = 0
        doc["materials"] = [
            {
                "name": "photo",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "extensions": {"KHR_materials_unlit": {}},
            }
        ]
        doc["textures"] = [{"sampler": 0, "source": 0}]
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        doc["images"] = [{"bufferView": image_view, "mimeType": texture_mime}]
        doc["extensionsUsed"] = ["KHR_materials_unlit"]
    else:
        rgba = np.empty((len(pos), 4), dtype=np.uint8)
        rgba[:, :3] = colors
        rgba[:, 3] = 255
        attributes["COLOR_0"] = add(
            rgba.tobytes(),
            ARRAY_BUFFER,
            {"componentType": UNSIGNED_BYTE, "normalized": True, "count": len(pos), "type": "VEC4"},
        )
    views, offset, body = [], 0, b""
    for blob, target in blobs:
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if target is not None:
            view["target"] = target
        views.append(view)
        padded = _pad(blob, b"\0")
        body += padded
        offset += len(padded)
    doc["meshes"] = [{"name": "surface", "primitives": [primitive]}]
    doc["accessors"] = accessors
    doc["bufferViews"] = views
    doc["buffers"] = [{"byteLength": len(body)}]
    js = _pad(json.dumps(doc, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(body)
    return b"".join(
        [
            struct.pack("<III", MAGIC, 2, total),
            struct.pack("<II", len(js), JSON_CHUNK),
            js,
            struct.pack("<II", len(body), BIN_CHUNK),
            body,
        ]
    )


@dataclass
class GlbMesh:
    doc: dict[str, Any]
    vertices: npt.NDArray[np.float32]
    faces: npt.NDArray[np.uint32]
    colors: npt.NDArray[np.uint8] | None
    """COLOR_0 as RGB, or None for a textured mesh."""
    uv: npt.NDArray[np.float32] | None = None
    texture: bytes | None = None


def read_mesh_glb(data: bytes) -> GlbMesh:
    """Reads files written by `write_mesh_glb` (tests and checks only)."""
    js_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + js_len])
    body = data[28 + js_len :]

    def raw(view_index: int) -> bytes:
        bv = doc["bufferViews"][view_index]
        return body[bv["byteOffset"] : bv["byteOffset"] + bv["byteLength"]]

    def view(i: int, dtype: type[np.generic], width: int) -> npt.NDArray[Any]:
        acc = doc["accessors"][i]
        return np.frombuffer(raw(acc["bufferView"]), dtype=dtype).reshape(acc["count"], width)

    prim = doc["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    return GlbMesh(
        doc,
        view(attrs["POSITION"], np.float32, 3),
        view(prim["indices"], np.uint32, 1).reshape(-1, 3),
        view(attrs["COLOR_0"], np.uint8, 4)[:, :3] if "COLOR_0" in attrs else None,
        view(attrs["TEXCOORD_0"], np.float32, 2) if "TEXCOORD_0" in attrs else None,
        raw(doc["images"][0]["bufferView"]) if doc.get("images") else None,
    )


@dataclass
class GlbScene:
    doc: dict[str, Any]
    points: npt.NDArray[np.float32]
    colors: npt.NDArray[np.uint8]
    path: npt.NDArray[np.float32] | None


def read_glb(data: bytes) -> GlbScene:
    """Reads files written by `write_glb` (used by tests and checks, not a general loader)."""
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or version != 2 or total != len(data):
        raise ValueError("Not a glTF 2.0 binary.")
    js_len, js_type = struct.unpack_from("<II", data, 12)
    if js_type != JSON_CHUNK:
        raise ValueError("First chunk is not JSON.")
    doc = json.loads(data[20 : 20 + js_len])
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + js_len)
    if bin_type != BIN_CHUNK:
        raise ValueError("Second chunk is not BIN.")
    body = data[28 + js_len : 28 + js_len + bin_len]

    def accessor(i: int, dtype: type[np.generic], width: int) -> npt.NDArray[Any]:
        acc = doc["accessors"][i]
        view = doc["bufferViews"][acc["bufferView"]]
        raw = body[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        return np.frombuffer(raw, dtype=dtype).reshape(acc["count"], width)

    pts = accessor(0, np.float32, 3)
    cols = accessor(1, np.uint8, 4)[:, :3]
    path = accessor(2, np.float32, 3) if len(doc["accessors"]) > 2 else None
    return GlbScene(doc, pts, cols, path)
