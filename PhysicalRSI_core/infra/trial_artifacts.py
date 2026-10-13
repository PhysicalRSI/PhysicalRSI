"""Copy adapter-owned files into a trial's verified evidence closure.

Adapters are trusted local code. This retains declared bytes, not external
assets referenced by those bytes, and does not certify their scientific origin.
"""
import os
from pathlib import Path
import shutil

from .storage import atomic_json, file_digest, read_json, relative_path


def _owned_path(folder, path):
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Trial artifact escaped its directory")
    current = path
    while current != folder:
        if current.is_symlink():
            raise ValueError("Trial artifact cannot traverse a symlink")
        current = current.parent


def artifact_entries(folder, manifest):
    """Validate the declared paths; the caller verifies their exact bytes."""
    if manifest.get("schema") != "physicalrsi.trial-artifacts/v1":
        raise ValueError("Unknown trial artifact schema")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("Trial artifact files must be a manifest")
    for name, sha in files.items():
        relative = relative_path(name)
        if relative.parts[0] != "artifacts" or len(relative.parts) < 2 or str(relative) != name:
            raise ValueError("Trial artifact escaped its namespace")
        path = folder / relative
        _owned_path(folder, path)
    return dict(files)


def artifact_names(folder):
    files = artifact_entries(folder, read_json(folder / "artifacts.json"))
    for name, sha in files.items():
        if file_digest(folder / name) != sha:
            raise ValueError("Trial artifact evidence changed: " + name)
    return {"artifacts.json", *files}


def capture(folder, exports):
    """exports maps relative names to {path, sha256}; never retain live links."""
    if not isinstance(exports, dict):
        raise ValueError("Adapter must export an artifact manifest")
    files = {}
    for name, entry in exports.items():
        relative = relative_path(name)
        if str(relative) != name or set(entry) != {"path", "sha256"}:
            raise ValueError("Invalid exported trial artifact")
        source = Path(entry["path"])
        if source.is_symlink() or not source.is_file():
            raise ValueError("Exported artifact must be a regular file")
        target = folder / "artifacts" / relative
        _owned_path(folder, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        sha = file_digest(target)
        if sha != entry["sha256"]:
            raise ValueError("Exported artifact changed during capture")
        files[str(target.relative_to(folder))] = sha
    directories = {parent for name in files for parent in (folder / name).parents
                   if parent == folder or parent.is_relative_to(folder)}
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    atomic_json(folder / "artifacts.json", dict(schema="physicalrsi.trial-artifacts/v1", files=files))
    return artifact_names(folder)
