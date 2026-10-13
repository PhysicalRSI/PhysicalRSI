from pathlib import Path

import pytest

from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import read_json
from PhysicalRSI_core.self_harness.archive import CandidateArchive
from PhysicalRSI_demos.staged_evolution import setup
from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.fidelity import MultiFidelityEvaluator
from PhysicalRSI_core.self_harness.protocol import PreregisteredSuite
from PhysicalRSI_demos.rollout_evolution import Suite


def test_screening_saves_full_trials_and_keeps_distinct_scores_and_archived_branches(tmp_path):
    measured = {}
    for screened,screen_incumbent in ((False,False),(True,True),(True,False)):
        root = tmp_path / (str(screened)+'-'+str(screen_incumbent))
        campaign, initial, suite, quota = setup(root, screened=screened,screen_incumbent=screen_incumbent,max_trials=26)
        result = campaign.run(initial)
        report = campaign.report_test(initial)
        assert [r["outcome"] for r in result["rounds"]] == ["inherit_child", "no_changed_candidate"]
        assert all(m["success_rate"] == 1 for m in report["metrics"].values())
        measured[screened,screen_incumbent] = len(suite.resets)
        assert len(suite.resets) == quota.status()["reserved_trials"]
        directory = campaign.root / "rounds/round-0001/execution"
        comparison = read_json(directory / "freeze.json")["output"]
        if screened:
            screening = read_json(directory / "screening.json")["output"]
            assert set(comparison["candidates"]) == {"uncalibrated", "calibrated-0"}
            baseline=screening['candidates']['uncalibrated']
            assert baseline['measured'] is screen_incumbent
            assert baseline['per_task']==({'axis':0.,'vector':0.} if screen_incumbent else None)
            assert (directory/'experiments/screening/development/experiments/development/uncalibrated').exists() is screen_incumbent
            assert screening["candidates"]["uncalibrated"]["accepted"]
            assert screening["evaluator_revision"] != read_json(directory / "config.json")["protocol"]["identity"]
            assert not (directory / "experiments/validation/overcorrection-2").exists()
            archive = CandidateArchive(root / "archive")
            entries = [archive.read(p.parent.name) for p in (archive.root / "entries").glob("*/record.json")]
            first = next(e for e in entries if "overcorrection-2" in e["candidates"])
            rejected = first["candidates"]["overcorrection-2"]
            assert first['candidates']['uncalibrated']['screening']['measured'] is screen_incumbent
            assert rejected["screening"]["per_task"] == {"axis": 0.0, "vector": 0.0}
            assert rejected["metrics"] is None and rejected["disposition"] == "admission_rejected"
            assert any(name.startswith("experiments/screening/") for name in first["files"])
            jobs = JobScheduler(root / "scheduler", capacities={"cpu": 2}).status()
            assert len(jobs["jobs"]) == (7 if screen_incumbent else 6) and jobs["used"] == {"cpu": 0}
        else:
            assert len(comparison["candidates"]) == 5
        reopened, seed, new_suite, _ = setup(root, screened=screened,screen_incumbent=screen_incumbent,max_trials=26)
        assert reopened.run(seed) == result and reopened.report_test(seed) == report
        assert new_suite.resets == []
    assert measured == {(False,False):26,(True,True):24,(True,False):22}


def test_whole_screening_pool_requires_quota_before_first_screening_effect(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path, max_trials=5)
    result = campaign.run(initial)
    assert result["status"] == "budget_exhausted"
    assert len(suite.resets) == quota.status()["reserved_trials"] == 2
    assert campaign.state.resolve()["harness"]["id"] == initial["id"]
    assert JobScheduler(tmp_path / "scheduler", capacities={"cpu": 2}).status()["jobs"] == {}


def test_tampered_screening_evidence_blocks_resume_and_archive_read(tmp_path):
    campaign, initial, suite, _ = setup(tmp_path)
    campaign.run(initial)
    directory = campaign.root / "rounds/round-0001/execution"
    screening = read_json(directory / "screening.json")["output"]
    name = next(name for name in screening["evidence"] if name.endswith("trajectory.json"))
    (directory / name).write_text("changed")
    before = len(suite.resets)
    with pytest.raises(ValueError):
        campaign.run(initial)
    assert len(suite.resets) == before
    archive = CandidateArchive(tmp_path / "archive")
    first = next(archive.read(p.parent.name) for p in (archive.root / "entries").glob("*/record.json")
                 if name in read_json(p)["files"])
    owned = archive.root / "entries" / first["revision"] / "evidence" / name
    owned.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        archive.read(first["revision"])


@pytest.mark.parametrize("cheap_outcome", ["success", "failure", "uncertain"])
def test_cheap_outcome_is_never_used_as_full_validation_score(tmp_path, monkeypatch, cheap_outcome):
    class CheapVerifier:
        def identity(self):
            return dict(kind="deliberately-inexact-screening-test", outcome=cheap_outcome)

        def verify(self, case, trace):
            return dict(outcome=cheap_outcome, reason="Explicitly weak screening test protocol", measurements={})

    class CheapSuite(Suite):
        def identity(self):
            return dict(super().identity(), cheap_verifier=CheapVerifier().identity())

        def verifier(self, task):
            return CheapVerifier()

    initialize = MultiFidelityEvaluator.__init__
    cheap_suites = []

    def different_screening(self, **kwargs):
        original = kwargs["screening"]
        suite = CheapSuite({"axis": 2., "vector": .5})
        cheap_suites.append(suite)
        kwargs["screening"] = ExperimentEvaluator(suite=PreregisteredSuite(suite, original.suite.protocol),
            budgets=original.budgets, scope=original.scope, system2=original.system2, quota=original.quota)
        initialize(self, **kwargs)

    monkeypatch.setattr(MultiFidelityEvaluator, "__init__", different_screening)
    campaign, initial, suite, quota = setup(tmp_path, max_trials=36)
    directory = campaign.root / "rounds/round-0001/execution"
    if cheap_outcome == "uncertain":
        with pytest.raises(ReconciliationRequired, match="no definite outcome"):
            campaign.run(initial)
        assert len(suite.resets) == 2
        assert sum(len(s.resets) for s in cheap_suites) == 8
        assert quota.status()["reserved_trials"] == 10
        assert not (directory / "validation.json").exists()
        assert campaign.state.resolve()["harness"]["id"] == initial["id"]
    elif cheap_outcome=='failure':
        result=campaign.run(initial)
        assert result['current']['harness']['id']==initial['id']
        assert [row['outcome'] for row in result['rounds']]==['all_candidates_rejected','all_candidates_rejected']
        assert not (directory/'validation.json').exists()
        parent=read_json(directory/'screening.json')['output']['candidates'][initial['id']]
        assert parent['measured'] is False and parent['per_task'] is None
    else:
        result = campaign.run(initial)
        assert result["current"]["harness"]["id"] == "calibrated-0"
        screening = read_json(directory / "screening.json")["output"]
        selection = read_json(directory / "selection.json")["output"]
        assert screening["candidates"]["overcorrection-2"]["per_task"] == {"axis": 1., "vector": 1.}
        assert selection["metrics"]["overcorrection-2"]["per_task"] == {"axis": 0., "vector": 0.}
    assert JobScheduler(tmp_path / "scheduler", capacities={"cpu": 2}).status()["used"] == {"cpu": 0}


@pytest.mark.parametrize('original',[False,True])
def test_incumbent_measurement_policy_is_frozen_before_restart(tmp_path,original):
    campaign,initial,_,quota=setup(tmp_path,screen_incumbent=original,max_trials=26)
    campaign.run(initial);before=quota.status()
    reopened,seed,suite,_=setup(tmp_path,screen_incumbent=not original,max_trials=26)
    with pytest.raises(ValueError,match='Frozen'):
        reopened.run(seed)
    assert not suite.resets and quota.status()==before
