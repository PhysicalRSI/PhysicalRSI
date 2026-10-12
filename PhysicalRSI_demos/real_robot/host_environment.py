"""Explicit host environment binding, with no SDK discovery or local paths."""
import hashlib
import json
from pathlib import Path
import platform
import sys

from .contracts import digest


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Environment file changed while hashing: " + str(path))
    return dict(path=str(path), sha256=sha, bytes=after.st_size)


def capture_environment(*, files, package_versions):
    """Host declares all config, SDK, kinematic assets and relevant packages.

    Bind this callable to the actual deployment inventory. The core cannot infer
    completeness, firmware identity, calibration or physical scene restoration.
    """
    if not isinstance(files, dict) or not files:
        raise ValueError("Explicit nonempty host file inventory required")
    if not isinstance(package_versions, dict) or not package_versions:
        raise ValueError("Explicit host package versions required")
    return dict(schema="tianji.physical-environment/v1",
                files={name: file_identity(path) for name, path in sorted(files.items())},
                python=dict(version=platform.python_version(), executable=file_identity(sys.executable)),
                package_versions=dict(package_versions),
                limitations=["Host-declared inventory; no firmware/OS/calibration attestation"])


def require_same_environment(expected, actual):
    if expected.get("schema") != "tianji.physical-environment/v1" or digest(expected) != digest(actual):
        raise ValueError("Physical software/configuration/assets changed; existing comparison qualification cannot transfer")


def verify_trial_environment(directory, expected, *, evidence=None):
    directory = Path(directory)
    for relative in ("evaluation/environment-before.json", "hardware-owner/environment.json",
                     "evaluation/environment-after.json"):
        path = directory / relative
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Environment receipts must be physical files")
        if evidence is not None and evidence.get(str(path)) != file_identity(path)["sha256"]:
            raise ValueError("Physical environment receipt is not part of the committed comparison evidence")
        require_same_environment(expected, json.loads(path.read_text()))
