"""Treat an exhausted, fully recorded LIBERO action allowance as termination.

Only the exact trusted primitive exhaustion at the declared boundary is handled.
Timeouts, solver failures and unrecorded native effects remain interruptions.
"""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest
from .recorded_cap import RecordedPrimitiveEnvironment


class BoundedLiberoEnvironment(RecordedPrimitiveEnvironment):
    def __init__(self, *, native_steps, **kwargs):
        if type(native_steps) is not int or native_steps < 1:
            raise ValueError("Declare a positive native step allowance")
        super().__init__(**kwargs)
        self.native_steps = native_steps

    def identity(self):
        return {**super().identity(), "native_steps": self.native_steps,
                "budget_termination_implementation": file_digest(Path(__file__))}

    def step(self, action, context):
        try:
            return super().step(action, context)
        except RuntimeError as error:
            if str(error) != "Primitive physics-step budget exhausted":
                raise
            trace = self._api.trace
            if (self._api.physics_steps != self.native_steps
                    or len(trace) != self.native_steps
                    or [row.get("step") for row in trace] != list(range(1, self.native_steps + 1))):
                raise
            context.check()
            observation = self._observation({"id": action["id"], "value": {
                "status": "native_budget_exhausted", "physics_steps": self.native_steps}})
            observation["evaluation"]["termination_reason"] = "native_budget_exhausted"
            context.check()
            self._finished = True
            return {"observation": observation, "terminated": True}
