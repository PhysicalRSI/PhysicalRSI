from copy import deepcopy

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.reseeded_ik_worker import NOMINAL_HOME, solve_with_fallback


def test_current_solution_is_preserved_without_extra_solve():
    seen = []
    def solve(request):
        seen.append(request)
        return {"joints": [1] * 7}
    result = solve_with_fallback({"joints": [0] * 7}, solve=solve)
    assert len(seen) == 1 and result["joints"] == [1] * 7
    assert result["seed_attempts"] == [{"seed": "current", "accepted": True}]


@pytest.mark.parametrize("error", [ValueError("Invalid IK solution"), RuntimeError("IK target did not converge")])
def test_known_rejection_retries_seed_without_changing_target_or_input(error):
    request = {"joints": [0] * 7, "pose": [1, 2, 3], "gripper_fraction": .9}
    original = deepcopy(request)
    seen = []
    def solve(value):
        seen.append(value)
        if len(seen) == 1:
            raise error
        assert value == {**original, "joints": list(NOMINAL_HOME)}
        return {"joints": [2] * 7}
    result = solve_with_fallback(request, solve=solve)
    assert request == original and len(seen) == 2
    assert result["seed_attempts"][0]["rejected"] == str(error)


def test_both_rejections_are_bounded_and_never_return_a_solution():
    seen = []
    def solve(value):
        seen.append(value)
        raise ValueError("Invalid IK solution")
    with pytest.raises(RuntimeError, match="IK target did not converge"):
        solve_with_fallback({"joints": [0] * 7}, solve=solve)
    assert len(seen) == 2


@pytest.mark.parametrize("error", [ValueError("Invalid IK inputs"), RuntimeError("Unexpected model"), TimeoutError("deadline")])
def test_unknown_errors_remain_fatal_without_retry(error):
    seen = []
    def solve(value):
        seen.append(value)
        raise error
    with pytest.raises(type(error), match=str(error)):
        solve_with_fallback({"joints": [0] * 7}, solve=solve)
    assert len(seen) == 1


def test_request_journal_uses_transport_order_after_id_is_removed(tmp_path, monkeypatch):
    import json
    from PhysicalRSI_baselines.embodied_goodharts_law import reseeded_ik_worker as module
    monkeypatch.setattr(module.sys, "argv", ["worker", str(tmp_path)])
    monkeypatch.setattr(module.worker, "_worker", module.worker._worker)
    calls = []
    def solve(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("IK target did not converge")
        return {"joints": [0] * 7}
    monkeypatch.setattr(module, "solve_with_fallback", solve)
    def transport():
        with pytest.raises(RuntimeError, match="did not converge"):
            module.worker._worker({"joints": [1] * 7})
        module.worker._worker({"joints": [2] * 7})
    monkeypatch.setattr(module.worker, "main", transport)
    module.main()
    assert json.loads((tmp_path / "request-0000.json").read_text())["id"] == 0
    assert json.loads((tmp_path / "request-0001.json").read_text())["id"] == 1
    assert not (tmp_path / "solution-0000.json").exists()
    assert (tmp_path / "solution-0001.json").exists()
