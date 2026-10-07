"""A separately identified persistent IK runtime with one bounded seed fallback."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest
from .persistent_aspire_ik import PersistentAspireHandIK


class ReseededAspireHandIK(PersistentAspireHandIK):
    def _command(self):
        return [self.interpreter, "-m",
                "PhysicalRSI_baselines.embodied_goodharts_law.reseeded_ik_worker",
                str(self.directory)]

    def identity(self):
        return {**super().identity(),
                "reseeded_client_sha256": file_digest(Path(__file__)),
                "reseeded_worker_sha256": file_digest(Path(__file__).with_name("reseeded_ik_worker.py")),
                "seed_policy": "current joints first; one nominal-home fallback after solver rejection",
                "collision_planning": False}
