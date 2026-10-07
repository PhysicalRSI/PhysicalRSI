"""Preserve native budget termination while releasing closed simulator callbacks."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest
from .bounded_libero_environment import BoundedLiberoEnvironment
from .released_recorded_environment import ReleasedRecordedEnvironment


class ReleasedBoundedLiberoEnvironment(BoundedLiberoEnvironment, ReleasedRecordedEnvironment):
    def identity(self):
        return {**super().identity(), "released_bounded_implementation_sha256": file_digest(Path(__file__))}
