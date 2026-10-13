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


def bounded_screening(monkeypatch, options):
    initialize = MultiFidelityEvaluator.__init__

    def bounded(self, **kwargs):
        kwargs.update(options)
        initialize(self, **kwargs)

    monkeypatch.setattr(MultiFidelityEvaluator, '__init__', bounded)


def ranking_profile():
    return {t: dict(weight=1, score_range=[0, 1]) for t in ('axis', 'vector')}


@pytest.mark.parametrize('screen_incumbent', [False, True])
def test_ranked_survivors_retain_incumbent_and_all_branch_evidence(tmp_path, monkeypatch, screen_incumbent):
    options = dict(survivor_limit=2, ranking_profile=ranking_profile(), minimum_success_rate=0.)
    bounded_screening(monkeypatch, options)
    campaign, initial, suite, quota = setup(tmp_path, screen_incumbent=screen_incumbent, max_trials=48)
    result = campaign.run(initial)
    report = campaign.report_test(initial)
    directory = campaign.root / 'rounds/round-0001/execution'
    screen = read_json(directory / 'screening.json')['output']
    rows = screen['candidates']
    ranked = screen['ranking']
    assert ranked[0] == 'calibrated-0' and len(ranked) == 4
    assert ranked[1:] == sorted(ranked[1:], key=lambda name: rows[name]['freeze_sha256'])
    assert [rows[name]['ranking']['rank'] for name in ranked] == [1, 2, 3, 4]
    assert rows[ranked[0]]['ranking']['per_task'] == {'axis': 1., 'vector': 1.}
    assert rows[initial['id']]['accepted'] and rows[initial['id']]['measured'] is screen_incumbent
    assert {name for name, row in rows.items() if row['accepted']} == {initial['id'], *ranked[:2]}
    frozen = read_json(directory / 'freeze.json')['output']
    assert set(frozen['candidates']) == {initial['id'], *ranked[:2]}
    assert all(not (directory / 'experiments/validation' / name).exists() for name in ranked[2:])
    assert result['current']['harness']['id'] == 'calibrated-0'
    assert all(v['success_rate'] == 1 for v in report['metrics'].values())
    archive = CandidateArchive(tmp_path / 'archive')
    entries = [archive.read(p.parent.name) for p in (archive.root / 'entries').glob('*/record.json')]
    first = next(e for e in entries if ranked[-1] in e['candidates'])
    for name in ranked[2:]:
        row = first['candidates'][name]
        assert row['disposition'] == 'admission_rejected' and row['metrics'] is None
        assert row['screening']['ranking'] == rows[name]['ranking']
        assert row['screening']['reason'] == 'development screening survivor limit'
    before = quota.status()
    reopened, seed, fresh, _ = setup(tmp_path, screen_incumbent=screen_incumbent, max_trials=48)
    assert reopened.run(seed) == result and reopened.report_test(seed) == report
    assert not fresh.resets and quota.status() == before
    options['survivor_limit'] = 1
    reopened, seed, fresh, _ = setup(tmp_path, screen_incumbent=screen_incumbent, max_trials=48)
    with pytest.raises(ValueError, match='Frozen'):
        reopened.run(seed)
    assert not fresh.resets and quota.status() == before


@pytest.mark.parametrize('options', [
    dict(survivor_limit=0, ranking_profile=ranking_profile()),
    dict(survivor_limit=True, ranking_profile=ranking_profile()),
    dict(survivor_limit=1.5, ranking_profile=ranking_profile()),
    dict(survivor_limit=1),
    dict(ranking_profile=ranking_profile()),
    dict(survivor_limit=1, ranking_profile={'axis': dict(weight=1, score_range=[0, 1])}),
    dict(survivor_limit=1, ranking_profile={t: dict(weight=True, score_range=[0, 1]) for t in ('axis', 'vector')}),
    dict(survivor_limit=1, ranking_profile={t: dict(weight=0, score_range=[0, 1]) for t in ('axis', 'vector')}),
    dict(survivor_limit=1, ranking_profile={t: dict(weight=1, score_range=[1, 1]) for t in ('axis', 'vector')}),
    dict(survivor_limit=1, ranking_profile={t: dict(weight=1, score_range=[-1e308, 1e308]) for t in ('axis', 'vector')}),
    dict(survivor_limit=1, ranking_profile={t: dict(weight=1e308, score_range=[0, 1]) for t in ('axis', 'vector')}),
])
def test_invalid_ranking_policy_is_rejected_before_trials(tmp_path, monkeypatch, options):
    bounded_screening(monkeypatch, options)
    campaign, initial, suite, quota = setup(tmp_path)
    with pytest.raises(ValueError):
        campaign.run(initial)
    assert not suite.resets and quota.status()['reserved_trials'] == 0


def test_ranking_normalizes_native_scales_and_task_weights_not_episode_counts():
    from types import SimpleNamespace
    evaluator = object.__new__(MultiFidelityEvaluator)
    evaluator.screening = SimpleNamespace(score_measurement='quality')
    evaluator.ranking_profile = {'axis': dict(weight=3, score_range=[-10, 10]),
                                 'vector': dict(weight=1, score_range=[0, 100])}
    def row(task, score):
        return dict(task=task, outcome='success', measurements=dict(quality=score))
    value = evaluator._ranking_scores([row('axis', 0), row('axis', 10), row('vector', 20)])
    assert value['per_task'] == {'axis': .75, 'vector': .2}
    assert value['score'] == pytest.approx(.6125)
    assert evaluator._ranking_scores([row('axis', 5), row('vector', 20)]) == value
    for bad in (None, True, float('inf'), float('nan'), 101):
        with pytest.raises(ValueError, match='score'):
            evaluator._ranking_scores([row('axis', 5), row('vector', bad)])


@pytest.mark.parametrize('measurement', ['squared_error', 'absent'])
def test_misleading_or_missing_ranking_measurements_cannot_promote(tmp_path, monkeypatch, measurement):
    initialize = MultiFidelityEvaluator.__init__

    def ranked(self, **kwargs):
        low = kwargs['screening']
        kwargs['screening'] = ExperimentEvaluator(suite=low.suite, budgets=low.budgets, scope=low.scope,
            system2=low.system2, quota=low.quota, score_measurement=measurement)
        # Deliberately reward worse squared error in development. Full validation
        # still requires actual success on independently generated paired cases.
        kwargs.update(survivor_limit=1, minimum_success_rate=0.,
            ranking_profile={t: dict(weight=1, score_range=[0, 1000]) for t in ('axis', 'vector')})
        initialize(self, **kwargs)

    monkeypatch.setattr(MultiFidelityEvaluator, '__init__', ranked)
    campaign, initial, suite, quota = setup(tmp_path, max_trials=64)
    directory = campaign.root / 'rounds/round-0001/execution'
    if measurement == 'absent':
        with pytest.raises(ValueError, match='score measurement'):
            campaign.run(initial)
        assert not (directory / 'validation.json').exists()
        assert quota.status()['reserved_trials'] == len(suite.resets) == 10
    else:
        result = campaign.run(initial)
        screen = read_json(directory / 'screening.json')['output']
        assert screen['ranking'][0] == 'overcorrection-4'
        assert set(read_json(directory / 'freeze.json')['output']['candidates']) == {'uncalibrated', 'overcorrection-4'}
        assert result['current']['harness']['id'] == initial['id']
    assert campaign.state.resolve()['harness']['id'] == initial['id']
    assert JobScheduler(tmp_path / 'scheduler', capacities={'cpu': 2}).status()['used'] == {'cpu': 0}


@pytest.mark.parametrize('score,repeats', [(.1, 11), (.9, 3)])
@pytest.mark.parametrize('repeated_task', ['axis', 'vector'])
def test_screening_repeated_constant_feedback_keeps_the_same_task_priority(score, repeats, repeated_task):
    from types import SimpleNamespace
    evaluator = object.__new__(MultiFidelityEvaluator)
    evaluator.screening = SimpleNamespace(score_measurement='quality')
    evaluator.ranking_profile = {'axis': dict(weight=3, score_range=[0, 1]),
                                 'vector': dict(weight=1, score_range=[0, 1])}
    other = 'vector' if repeated_task == 'axis' else 'axis'
    repeated = dict(task=repeated_task, outcome='success', measurements=dict(quality=score))
    fixed = dict(task=other, outcome='success', measurements=dict(quality=.2))
    expected = evaluator._ranking_scores([repeated, fixed])
    assert expected['per_task'][repeated_task] == score
    actual = evaluator._ranking_scores([repeated] * repeats + [fixed])
    assert actual == expected
    assert evaluator._ranking_scores([fixed] + [repeated] * repeats) == expected


def test_screening_slots_follow_content_when_candidates_are_renamed_and_reordered(tmp_path, monkeypatch):
    from PhysicalRSI_core.self_harness.artifacts import verify_harness
    from PhysicalRSI_demos.staged_evolution import CalibrationSearch

    bounded_screening(monkeypatch, dict(survivor_limit=2, ranking_profile=ranking_profile(),
                                       minimum_success_rate=0.))
    propose = CalibrationSearch.propose

    def renamed(self, *args):
        children = propose(self, *args)
        return [dict(child, id='renamed-' + str(len(children) - index))
                for index, child in reversed(list(enumerate(children)))]

    observed = []
    for variant in ('original', 'renamed'):
        if variant == 'renamed':
            monkeypatch.setattr(CalibrationSearch, 'propose', renamed)
        root = tmp_path / variant
        campaign, initial, _, quota = setup(root, max_trials=48)
        result = campaign.run(initial)
        report = campaign.report_test(initial)
        directory = campaign.root / 'rounds/round-0001/execution'
        screen = read_json(directory / 'screening.json')['output']
        admissions = screen['candidates']
        ranked = screen['ranking']
        assert len(ranked) == 4
        assert len({admissions[name]['ranking']['score'] for name in ranked[1:]}) == 1
        assert sum(admissions[name]['accepted'] for name in ranked) == 2
        assert admissions[initial['id']]['accepted']
        assert all(v['success_rate'] == 1 for v in report['metrics'].values())
        if variant == 'renamed':
            assert all(name.startswith('renamed-') for name in ranked)
        observed.append(dict(
            ranked=[admissions[name]['freeze_sha256'] for name in ranked],
            admitted=sorted(row['freeze_sha256'] for row in admissions.values() if row['accepted']),
            selected=verify_harness(result['current']['harness']),
            reserved_trials=quota.status()['reserved_trials']))
        # The reopened run consumes no additional screening or validation slots.
        reopened, seed, fresh, _ = setup(root, max_trials=48)
        assert reopened.run(seed) == result and reopened.report_test(seed) == report
        assert not fresh.resets and quota.status()['reserved_trials'] == observed[-1]['reserved_trials']
    assert observed[0] == observed[1]
