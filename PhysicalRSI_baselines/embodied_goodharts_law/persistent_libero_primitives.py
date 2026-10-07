"""Expose a pure persistent-IK miss as data, while preserving effect failures."""
from .located_libero_primitives import LocatedLiberoPrimitives
from .persistent_aspire_ik import IKNonConvergence


class PersistentIKLiberoPrimitives(LocatedLiberoPrimitives):
    def try_move_to_pose(self, pose):
        before = self.physics_steps
        try:
            return super().try_move_to_pose(pose)
        except IKNonConvergence:
            if self.physics_steps != before:
                raise RuntimeError("IK failure reported after native effects")
            return {"reached": False, "reason": "IK target did not converge",
                    "native_effects": False, "physics_steps": self.physics_steps,
                    "robot_state": self.get_robot_state()}
