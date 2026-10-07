"""A complete action ledger distinguishes budget termination from interruption."""
import pytest

from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.embodiment import Embodiment
from PhysicalRSI_baselines.embodied_goodharts_law.bounded_libero_environment import BoundedLiberoEnvironment


def environment(*, environment_class=BoundedLiberoEnvironment, steps=3, trace=None, error="Primitive physics-step budget exhausted"):
    class API:
        physics_steps = steps

        def __init__(self):
            self.trace = trace if trace is not None else [{"step": i} for i in range(1, 4)]

        @property
        def handlers(self):
            def move(args, kwargs, *, deadline):
                raise RuntimeError(error)
            return {"move": move}

    return environment_class(native_steps=3,
        specification=Embodiment("test", "1", "simulation", Contract("obs"), Contract("act")),
        identity={"test": True}, factory=lambda case: (API(), lambda: {"official_success": False}, lambda: None))


def invoke(env):
    from time import monotonic
    context = Context("trial", deadline=monotonic() + 10)
    env.reset({}, context)
    return env.step({"kind": "call", "id": 0, "method": "move", "args": [], "kwargs": {}}, context)


def test_complete_native_budget_terminates_without_claiming_success():
    result = invoke(environment())
    assert result["terminated"]
    assert not result["observation"]["evaluation"]["official_success"]
    assert result["observation"]["evaluation"]["termination_reason"] == "native_budget_exhausted"


@pytest.mark.parametrize("kwargs", [
    {"steps": 2}, {"trace": [{"step": 1}, {"step": 3}, {"step": 3}]},
    {"error": "ASPIRE IK worker failed"}, {"trace": []},
])
def test_incomplete_or_unrelated_failures_still_interrupt(kwargs):
    with pytest.raises(RuntimeError):
        invoke(environment(**kwargs))


def test_released_wrapper_preserves_budget_termination_and_drops_callbacks():
    from PhysicalRSI_baselines.embodied_goodharts_law.released_bounded_libero_environment import ReleasedBoundedLiberoEnvironment
    env = environment(environment_class=ReleasedBoundedLiberoEnvironment)
    result = invoke(env)
    assert result["terminated"]
    assert not result["observation"]["evaluation"]["official_success"]
    identity = env.identity()
    assert "budget_termination_implementation" in identity
    assert "release_implementation_sha256" in identity
    env.close()
    assert env._api is None and env._measure is None and not env._handlers
    env.close()
    broken = environment(environment_class=ReleasedBoundedLiberoEnvironment, steps=2)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        invoke(broken)
    broken.close()
