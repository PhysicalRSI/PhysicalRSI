from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


class Rejected(ValueError):
    """Rejected before an actuator effect; caller may use a NEW observation."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def source_roots():
    """Resolve executing packages; Core and GPT-as-Policy are not vendored here."""
    import PhysicalRSI_core
    from hybrid_rollout import robodojo
    return dict(adapter=Path(__file__).resolve().parent,
                physicalrsi_core=Path(PhysicalRSI_core.__file__).resolve().parent,
                gpt_as_policy=Path(robodojo.__file__).resolve().parent)


def executing_source_manifest(*, include_core=True):
    return {name + "/" + str(p.relative_to(root)): file_hash(p)
            for name, root in source_roots().items()
            if include_core or name != "physicalrsi_core"
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".md"}}


def implementation_hash():
    """Bind every adapter/skill file and the patched Galbot runtime to trials."""
    return digest(executing_source_manifest(include_core=False))
