"""Content addressed sensor blobs used by trial observations.

Blob references are deliberately small JSON values.  The bytes live below a
trial's ``blobs`` directory and are never embedded in the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import tempfile


_REF_KEYS = frozenset(("$blob", "media_type", "shape", "dtype", "bytes"))


def _valid_sha(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validate_shape(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Blob shape must be a list or null")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError("Blob shape entries must be nonnegative integers")
    return list(value)


def _validate_ref(value) -> dict:
    if isinstance(value, BlobRef):
        value = value.to_dict()
    if not isinstance(value, dict) or set(value) != _REF_KEYS:
        raise ValueError("Invalid blob reference")
    if not _valid_sha(value["$blob"]):
        raise ValueError("Blob reference digest must be lowercase SHA-256 hex")
    if not isinstance(value["media_type"], str) or not value["media_type"]:
        raise ValueError("Blob media_type must be a nonempty string")
    shape = _validate_shape(value["shape"])
    if value["dtype"] is not None and not isinstance(value["dtype"], str):
        raise ValueError("Blob dtype must be a string or null")
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("Blob bytes must be a nonnegative integer")
    return {
        "$blob": value["$blob"],
        "media_type": value["media_type"],
        "shape": shape,
        "dtype": value["dtype"],
        "bytes": value["bytes"],
    }


@dataclass(frozen=True)
class BlobRef:
    """A validated, JSON-compatible content address and its metadata."""

    sha256: str
    media_type: str
    shape: list[int] | None = None
    dtype: str | None = None
    bytes: int = 0

    def __post_init__(self):
        value = _validate_ref({
            "$blob": self.sha256,
            "media_type": self.media_type,
            "shape": self.shape,
            "dtype": self.dtype,
            "bytes": self.bytes,
        })
        object.__setattr__(self, "sha256", value["$blob"])
        object.__setattr__(self, "media_type", value["media_type"])
        object.__setattr__(self, "shape", value["shape"])
        object.__setattr__(self, "dtype", value["dtype"])
        object.__setattr__(self, "bytes", value["bytes"])

    def to_dict(self) -> dict:
        return {
            "$blob": self.sha256,
            "media_type": self.media_type,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "bytes": self.bytes,
        }

    @property
    def digest(self) -> str:
        return self.sha256

    # A JSON-oriented alias is convenient for adapters that use explicit names.
    as_dict = to_dict


def is_blob_ref(value) -> bool:
    """Return whether *value* is a complete, valid blob reference."""

    try:
        _validate_ref(value)
    except (TypeError, ValueError):
        return False
    return True


def refs(value):
    """Yield every blob reference nested in a JSON value.
    """

    if isinstance(value, BlobRef):
        yield value.to_dict()
    elif isinstance(value, dict):
        if is_blob_ref(value):
            yield _validate_ref(value)
        else:
            for item in value.values():
                yield from refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from refs(item)


class BlobStore:
    """A per-trial SHA-256 content addressed byte store."""

    def __init__(self, root, *, fsync=True):
        self.root = Path(root)
        self.fsync = bool(fsync)

    def ensure_root(self):
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _path(self, ref_or_sha):
        sha = ref_or_sha["$blob"] if isinstance(ref_or_sha, dict) else (
            ref_or_sha.sha256 if isinstance(ref_or_sha, BlobRef) else ref_or_sha)
        if not _valid_sha(sha):
            raise ValueError("Invalid blob digest")
        return self.root / sha[:2] / sha

    def path(self, ref) -> Path:
        """Return the store path for a validated reference."""

        value = _validate_ref(ref)
        return self._path(value)

    def put_bytes(self, data: bytes, *, media_type, shape=None, dtype=None, fsync=None) -> dict:
        if not isinstance(data, bytes):
            raise TypeError("Blob data must be bytes")
        value = _validate_ref({
            "$blob": self._digest(data),
            "media_type": media_type,
            "shape": shape,
            "dtype": dtype,
            "bytes": len(data),
        })
        target = self._path(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file() or target.stat().st_size != len(data):
                raise ValueError("Existing blob is corrupt")
            # The caller already has the bytes, so this check is cheap enough
            # at the deduplication boundary and prevents accepting a damaged CAS.
            if self._digest(target.read_bytes()) != value["$blob"]:
                raise ValueError("Existing blob is corrupt")
            return value
        fd, temporary = tempfile.mkstemp(prefix=".blob-", dir=target.parent)
        sync = self.fsync if fsync is None else bool(fsync)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                if sync:
                    os.fsync(stream.fileno())
            try:
                # A hard-link publish is atomic and never overwrites a blob
                # another producer has already installed.
                os.link(temporary, target)
            except FileExistsError:
                # Another producer won the race.  Validate its bytes before
                # returning the shared reference.
                if (target.is_symlink() or not target.is_file()
                        or target.stat().st_size != len(data)
                        or self._digest(target.read_bytes()) != value["$blob"]):
                    raise ValueError("Existing blob is corrupt")
            if sync:
                directory = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return value

    def put_array(self, array) -> dict:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised without numpy
            raise ImportError("BlobStore.put_array requires numpy") from exc
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return self.put_bytes(buffer.getvalue(), media_type="application/x-npy",
                              shape=[int(item) for item in array.shape], dtype=str(array.dtype))

    def put_png(self, array) -> dict:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("BlobStore.put_png requires numpy and Pillow") from exc
        value = np.asarray(array)
        if value.ndim not in (2, 3):
            raise ValueError("PNG arrays must be 2D or 3D")
        output = io.BytesIO()
        Image.fromarray(value).save(output, format="PNG")
        return self.put_bytes(output.getvalue(), media_type="image/png",
                              shape=[int(item) for item in value.shape], dtype=str(value.dtype))

    def _validated_path(self, ref):
        value = _validate_ref(ref)
        path = self._path(value)
        return value, path

    def stat(self, ref):
        """Return the referenced file's stat without reading its contents."""

        value, path = self._validated_path(ref)
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Referenced blob is missing: {value['$blob']}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError("Referenced blob is not a regular file")
        if info.st_size != value["bytes"]:
            raise ValueError("Referenced blob size differs from its reference")
        return info

    def check_ref(self, ref):
        """Check existence and declared size, without hashing the blob."""

        self.stat(ref)
        return ref

    def get_bytes(self, ref) -> bytes:
        value, _ = self._validated_path(ref)
        self.stat(value)
        data = self.path(value).read_bytes()
        if self._digest(data) != value["$blob"]:
            raise ValueError("Referenced blob digest differs from its reference")
        return data

    def verify_ref(self, ref) -> str:
        data = self.get_bytes(ref)
        return self._digest(data)

    def get_array(self, ref):
        import numpy as np
        value = _validate_ref(ref)
        payload = self.get_bytes(value)
        if value["media_type"] == "application/x-npy":
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        elif value["media_type"] == "image/png":
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Reading PNG blobs requires Pillow") from exc
            with Image.open(io.BytesIO(payload)) as image:
                array = np.asarray(image)
        else:
            raise ValueError("Blob is not a supported array media type")
        if value["shape"] is not None and list(array.shape) != value["shape"]:
            raise ValueError("Array shape differs from its blob reference")
        if value["dtype"] is not None and str(array.dtype) != value["dtype"]:
            raise ValueError("Array dtype differs from its blob reference")
        return array

    def files(self):
        if not self.root.exists():
            return []
        result = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or len(directory.name) != 2:
                continue
            for path in directory.iterdir():
                if path.is_file() and not path.is_symlink() and len(path.name) == 64:
                    result.append(path)
        return sorted(result)

    def unreferenced(self, referenced) -> list[str]:
        used = {_validate_ref(ref)["$blob"] for ref in referenced}
        return sorted(str(path.relative_to(self.root.parent)) for path in self.files()
                      if path.name not in used)
