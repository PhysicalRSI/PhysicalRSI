import multiprocessing

import pytest

from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded, TrialQuota


def reserve_in_process(root, name, queue):
    quota = TrialQuota(root, max_trials=3)
    try:
        quota.reserve({name + "-1": {"target": 1}, name + "-2": {"target": 2}})
        queue.put("admitted")
    except TrialLimitExceeded:
        queue.put("exhausted")


def test_reservations_are_atomic_across_processes(tmp_path):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    workers = [ctx.Process(target=reserve_in_process, args=(tmp_path, name, queue)) for name in ("a", "b")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0
    assert sorted(queue.get(timeout=1) for _ in workers) == ["admitted", "exhausted"]
    assert TrialQuota(tmp_path, max_trials=3).status()["reserved_trials"] == 2


def test_replay_is_free_but_cannot_change_reserved_inputs_or_allowance(tmp_path):
    quota = TrialQuota(tmp_path, max_trials=2)
    receipt = quota.reserve({"one": {"candidate": "parent"}})
    quota.reserve({"two": {"candidate": "child"}})
    assert quota.reserve({"one": {"candidate": "parent"}}) == receipt
    with pytest.raises(ValueError, match="inputs changed"):
        quota.reserve({"one": {"candidate": "different"}})
    with pytest.raises(ValueError, match="allowance changed"):
        TrialQuota(tmp_path, max_trials=3)
    with pytest.raises(TrialLimitExceeded):
        quota.reserve({"three": {"candidate": "new"}})
    assert quota.status() == dict(max_trials=2, reserved_trials=2, remaining_trials=0)
