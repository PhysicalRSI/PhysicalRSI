"""Small record storage for the extensible core.

The historical CandidateArchive remains available for audited Self-Harness
rounds.  CoreArchive only stores and retrieves values; callers decide what a
record means and whether it should be immutable.
"""

from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, read_json


class CoreArchive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, name, value):
        path = self.root / str(name)
        atomic_json(path, value)
        return path

    def get(self, name):
        return read_json(self.root / str(name))

    def names(self):
        return tuple(sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*.json")
        ))


__all__ = ["CoreArchive"]
