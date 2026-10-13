import multiprocessing

import pytest

from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded, TrialQuota


@pytest.mark.parametrize('value', [True, 1.0])
@pytest.mark.parametrize('entry', ['reopen', 'reserve'])
def test_quota_rejects_stored_allowance_type_drift(tmp_path, value, entry):
    from PhysicalRSI_core.infra.storage import atomic_json, read_json
    quota = TrialQuota(tmp_path, max_trials=1)
    path = tmp_path / 'reservations.json'
    saved = read_json(path)
    atomic_json(path, dict(saved, max_trials=value))
    before = path.read_bytes()
    request = {'trial': {'input': 1}}
    with pytest.raises(ValueError, match='Frozen trial allowance changed'):
        TrialQuota(tmp_path, max_trials=1) if entry == 'reopen' else quota.reserve(request)
    assert path.read_bytes() == before
    atomic_json(path, saved)
    receipt = quota.reserve(request)
    assert quota.reserve(request) == receipt
    assert quota.status()['reserved_trials'] == 1
    with pytest.raises(TrialLimitExceeded):
        quota.reserve({'another': {'input': 2}})


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
