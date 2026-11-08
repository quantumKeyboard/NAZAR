"""Local filesystem layout. Paths are built only from server-generated ids, never from
user-supplied names, so uploads cannot escape the storage root."""

import hashlib
import shutil
from pathlib import Path, PurePath

from nazar_api.settings import Settings

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v"})
UPLOAD_PART = "upload.part"
PREFLIGHT_DIR = "preflight"


def video_extension(filename: str) -> str | None:
    suffix = PurePath(filename).suffix.lower()
    return suffix if suffix in VIDEO_EXTENSIONS else None


def mission_dir(settings: Settings, mission_id: str) -> Path:
    return settings.storage_path / "missions" / mission_id


def asset_dir(settings: Settings, mission_id: str, asset_id: str) -> Path:
    return mission_dir(settings, mission_id) / "assets" / asset_id


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def job_frames_dir(settings: Settings, job_id: str) -> Path:
    """The frames a job was sampled to (`preflight.json` + `keyframes/NNNN.jpg`)."""
    return settings.storage_path / "jobs" / job_id / "frames"


def artifact_path(settings: Settings, job_id: str, artifact_id: str) -> Path:
    return settings.storage_path / "jobs" / job_id / "artifacts" / artifact_id
