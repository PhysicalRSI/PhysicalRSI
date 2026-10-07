import gc
import weakref

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.released_recorded_environment import ReleasedRecordedEnvironment


def test_closed_environment_does_not_keep_simulator_alive_through_callbacks():
    class Simulator:
        pass
    simulator = Simulator()
    reference = weakref.ref(simulator)
    environment = object.__new__(ReleasedRecordedEnvironment)
    environment._api = simulator
    environment._measure = lambda sim=simulator: sim
    environment._handlers = {"observe": lambda sim=simulator: sim}
    environment._close = lambda sim=simulator: None
    del simulator
    environment.close()
    environment.close()
    gc.collect()
    assert reference() is None
    assert environment._finished


def test_teardown_failure_is_not_hidden_or_reported_as_released():
    environment = object.__new__(ReleasedRecordedEnvironment)
    marker = object()
    environment._api = marker
    def fail():
        raise RuntimeError("native teardown failed")
    environment._close = fail
    for _ in range(2):
        with pytest.raises(RuntimeError, match="native teardown failed"):
            environment.close()
    assert environment._api is marker
