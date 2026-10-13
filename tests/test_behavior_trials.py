import pytest

from PhysicalRSI_baselines.behavior_baseline.trials import NativeTrials
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import atomic_json
from PhysicalRSI_core.infra.trial_quota import TrialLimitExceeded, TrialQuota


def requests():
    origin = System2Trial(Revision("fixture", "1"), "parent", "a" * 64, "development")
    return [dict(task="radio", instance_index=i, system2=origin) for i in (0, 1)]


def test_complete_plan_is_charged_atomically_and_cannot_change(tmp_path):
    quota = TrialQuota(tmp_path / "small-quota", max_trials=1)
    with pytest.raises(TrialLimitExceeded):
        NativeTrials(tmp_path, tmp_path / "rejected", trials=requests(), quota=quota)
    assert quota.status()["reserved_trials"] == 0
    assert not (tmp_path / "rejected/plan.json").exists()
    quota = TrialQuota(tmp_path / "quota", max_trials=2)
    plan = NativeTrials(tmp_path, tmp_path / "run", trials=requests(), quota=quota)
    assert quota.status()["reserved_trials"] == 2
    assert NativeTrials(tmp_path, tmp_path / "run", trials=requests(), quota=quota).plan == plan.plan
    changed = requests()
    changed[0]["seconds"] = 10
    with pytest.raises(ValueError, match="plan changed"):
        NativeTrials(tmp_path, tmp_path / "run", trials=changed, quota=quota)
    duplicate = [requests()[0], dict(requests()[0], port=8001)]
    with pytest.raises(ValueError, match="Repeated trial"):
        NativeTrials(tmp_path, tmp_path / "duplicate", trials=duplicate, quota=quota)


def test_unresolved_trial_blocks_starting_another_case(tmp_path, monkeypatch):
    from PhysicalRSI_baselines.behavior_baseline import native
    quota = TrialQuota(tmp_path / "quota", max_trials=2)
    plan = NativeTrials(tmp_path, tmp_path / "run", trials=requests(), quota=quota)
    first = plan.output / "trials" / plan.plan["trials"][0]["id"]
    atomic_json(first / "receipt.json", dict(state="started", pid=123))
    monkeypatch.setattr(native, "run", lambda *a, **k: pytest.fail("Must reconcile the earlier trial"))
    with pytest.raises(ReconciliationRequired):
        plan.run(1)
