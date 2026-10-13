import pytest

from PhysicalRSI_core.lineage import HarnessState, StateConflict
from test_core_versions import seed


def test_rollback_retry_must_resolve_to_the_same_requested_target(tmp_path):
    state = HarnessState(tmp_path / "state")
    initial = state.initialize(seed(tmp_path / "source"), policy={}, scope="software")
    body = {key: value for key, value in initial.items() if key != "revision"}
    second = state._publish(dict(body, previous=initial["revision"], action="selection"))
    third = state._publish(dict(body, previous=second["revision"], action="selection"))
    restored = state.rollback(third["revision"], target=initial["revision"])
    assert state.rollback(third["revision"], target=initial["revision"]) == restored
    # Omitting target would have requested the second generation, so it must
    # not be mistaken for a retry of the explicit jump to the first generation.
    with pytest.raises(StateConflict):
        state.rollback(third["revision"])
    assert state.resolve() == restored
