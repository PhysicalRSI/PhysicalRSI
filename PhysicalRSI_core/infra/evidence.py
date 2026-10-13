"""One read-only audit of a file closure, without a cross-call cache.

JSON is parsed from exactly the bytes that are hashed. Large binary evidence is
streamed and not retained. Create a reader inside one audit, never across adapter
calls or execution boundaries; the returned documents describe checked bytes,
not a lock on the live filesystem. A subsequent audit reads all files again.
"""
from hashlib import sha256
import json
from pathlib import Path

from .storage import file_digest, relative_path


class EvidenceReader:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if self.root != self.root.resolve():
            raise ValueError("Evidence root must be a physical directory")
        self.documents, self.hashes = {}, {}

    def path(self, name):
        relative = relative_path(name)
        if str(relative) != name:
            raise ValueError("Evidence path must be canonical")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Evidence escaped its directory")
        return path

    def document(self, name, expected=None):
        if name not in self.documents:
            payload = self.path(name).read_bytes()
            self.hashes[name] = sha256(payload).hexdigest()
            if expected is not None and self.hashes[name] != expected:
                raise ValueError("Experiment evidence changed: " + name)
            self.documents[name] = json.loads(payload)
        if expected is not None and self.hashes[name] != expected:
            raise ValueError("Experiment evidence changed: " + name)
        return self.documents[name]

    def verify(self, expected):
        for name, sha in expected.items():
            if name not in self.hashes:
                self.hashes[name] = file_digest(self.path(name))
            if self.hashes[name] != sha:
                raise ValueError("Experiment evidence changed: " + name)

    def unchanged_document(self, name):
        """Recheck a publication marker after reading its dependent files."""
        if file_digest(self.path(name)) != self.hashes[name]:
            raise ValueError("Experiment evidence changed during audit: " + name)
