"""Persist large/binary observations separately from compact execution records."""

import hashlib
import os
import stat
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path


class Artifacts:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, reference, *, max_bytes):
        """Read one local regular artifact with an explicit byte allowance.

        A reference is not authority to read arbitrary paths. The configured
        store and its ancestors must be protected from untrusted host writers.
        """
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Declare a nonnegative integer artifact byte allowance")
        if not isinstance(reference, dict) or set(reference) != {"artifact", "sha256", "bytes"}:
            raise ValueError("Invalid artifact reference")
        sha, size, name = reference["sha256"], reference["bytes"], reference["artifact"]
        if (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or type(size) is not int or not 0 <= size <= max_bytes or not isinstance(name, str)):
            raise ValueError("Invalid artifact digest, size or byte allowance")
        path = Path(name)
        if (str(path) != name or path.parent != self.root or path.name not in {sha + ".bin", sha + ".npy"}):
            raise ValueError("Artifact reference is outside this content-addressed store")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Artifact path must name a regular file")
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
                raise ValueError("Artifact is not a regular file of the declared size")
            content = stream.read(size + 1)
        if len(content) != size or hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Artifact content differs from its reference")
        return content

    def read_array(self, reference, *, max_bytes):
        """Read a non-object NPY v1/v2 array without trusting allocation sizes."""
        import io
        import math
        import numpy as np

        content = self.read(reference, max_bytes=max_bytes)
        if not reference["artifact"].endswith(".npy"):
            raise ValueError("Array artifact requires the NPY suffix")
        stream = io.BytesIO(content)
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError("Only NPY versions 1 and 2 are supported")
        shape, fortran, dtype = readers[version](stream, max_header_size=10000)
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("Array dimensions must be nonnegative integers")
        if dtype.hasobject or dtype.itemsize <= 0:
            raise ValueError("Array artifacts require fixed-size non-object data")
        count = math.prod(shape)
        if count * dtype.itemsize != len(content) - stream.tell():
            raise ValueError("Array shape and dtype do not match the stored byte count")
        # The immutable verified bytes own this view; no allocation follows an
        # unverified header and no pickle loader is invoked.
        return np.frombuffer(content, dtype=dtype, count=count, offset=stream.tell()).reshape(
            shape, order="F" if fortran else "C")

    def encode(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return self.encode(asdict(value))
        if isinstance(value, bytes):
            return self._bytes(value, ".bin")
        if isinstance(value, dict):
            return {str(key): self.encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.encode(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if type(value).__module__.split(".")[0] == "numpy":
            import io

            import numpy as np

            if isinstance(value, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                return self._bytes(buffer.getvalue(), ".npy")
            return value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported evidence type: {type(value).__name__}")

    def _bytes(self, content, suffix):
        sha = hashlib.sha256(content).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (sha + suffix)
        reference = {"artifact": str(path), "sha256": sha, "bytes": len(content)}
        if path.exists() or path.is_symlink():
            self.read(reference, max_bytes=len(content))
        else:
            fd, temporary = tempfile.mkstemp(dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)  # Publish without replacing existing evidence.
                except FileExistsError:
                    self.read(reference, max_bytes=len(content))
            finally:
                Path(temporary).unlink(missing_ok=True)
        directory = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return reference
