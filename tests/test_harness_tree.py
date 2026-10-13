from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from PhysicalRSI_core.contracts import Context, Contract, ReconciliationRequired
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.harness_tree import HarnessTree, RoutedPolicy
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.lineage import HarnessState, StateConflict
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, EvolutionSpec, verify_harness
from PhysicalRSI_core.self_harness.retention import RetentionSelector
from PhysicalRSI_core.self_harness.selection import select_survivor
from PhysicalRSI_core.trials import Evaluation, Trial
from PhysicalRSI_demos.harness_tree import Backend, PublicCapabilities, Suite, setup
from PhysicalRSI_demos.recursive_research import manifest


def leaf(root, coefficient=1., *, name="leaf"):
    for kind in CLOSURE:
        atomic_json(root / (kind + ".json"), dict(coefficient=coefficient) if kind == "control" else dict(kind=kind))
    return manifest(root, name)


def tree(root, *, nested=True):
    nodes = {"fallback":dict(parent=None,when={},harness=leaf(root / "source/base"))}
    if nested:
        nodes["shifted"] = dict(parent="fallback",when={"drive":"shifted"},harness=leaf(root / "source/shifted",.5))
        nodes["vector"] = dict(parent="shifted",when={"drive":"shifted","action_contract":"vector"},
                               harness=leaf(root / "source/vector",2.))
    return HarnessTree.create(nodes, keys=["action_contract","drive","task"], output=root / "package",name="tree",
                              evolution=EvolutionSpec(("control","skill_selection")))


def test_tree_learning_preserves_old_capabilities_and_rejects_weighted_forgetting(tmp_path):
    campaign, initial, suite, quota = setup(tmp_path)
    research = campaign.run(initial)
    final = campaign.report_test(initial)
    assert [r["outcome"] for r in research["rounds"]] == ["inherit_child","inherit_child","no_changed_candidate"]
    selected = HarnessTree(research["current"]["harness"])
    assert len(selected.body["nodes"]) == 3
    assert all(v["successes"] == v["planned"] == 2 for v in final["metrics"].values())
    directory = tmp_path / "campaign/rounds/round-0002/execution"
    decision = read_json(directory / "selection.json")["output"]
    metrics = decision["metrics"]
    assert metrics["global-repair"]["normalized_score"] > metrics[decision["parent_id"]]["normalized_score"]
    assert decision["rejected"]["global-repair"]["reason"] == "historical_capability_regression"
    assert set(decision["retention"]["violations"]["global-repair"]) == {"axis-default","vector-default","axis-shifted"}
    comparison, cohort = (read_json(directory / (name + ".json"))["output"] for name in ("freeze","validation"))
    results = [read_json(directory / ("evaluate_" + name + ".json"))["output"] for name in comparison["candidates"]]
    assert "global-repair" not in select_survivor(comparison,cohort,results,evidence_root=directory)["rejected"]
    routes = [read_json(p) for p in (tmp_path / "campaign/final-test").rglob("policy-evidence.json")]
    assert len(routes) == 8
    for route in routes:
        assert route["tree_revision"] == selected.revision
        assert route["path"] == selected.route(route["capabilities"])
        if route["capabilities"]["drive"] == "unseen-standard":
            assert route["path"] == ["fallback"]
    assert quota.status()["reserved_trials"] == len(suite.resets) == 60
    restarted, again, observed, _ = setup(tmp_path)
    assert restarted.run(again) == research and restarted.report_test(again) == final
    assert not observed.resets
    # The selected tree uses normal lineage rollback, including all branches.
    limits = {task: 0. for task in final['metrics']}
    prior = RetentionSelector(state=campaign.state, parent_revision=research['current']['revision'],
                              maximum_regression=limits)
    assert prior.view['high_water'] == {task: 1. for task in limits}
    restored = campaign.state.rollback(research["current"]["revision"],reason="tree rollback test")
    assert len(HarnessTree(restored["harness"]).body["nodes"]) == 2
    retained = RetentionSelector(state=campaign.state, parent_revision=restored['revision'],
                                 maximum_regression=limits)
    assert retained.view['high_water'] == prior.view['high_water']
    assert retained.view['records'] == [restored['revision'], *prior.view['records']]
    assert prior.identity()['view'] == prior.view
    with pytest.raises(ValueError, match='frozen parent'):
        retained.select(comparison, cohort, results, evidence_root=directory)
    assert campaign.state.resolve() == restored
    assert quota.status()['reserved_trials'] == len(suite.resets) == 60


def test_owned_tree_reconstructs_logical_files_after_source_loss(tmp_path):
    candidate = tree(tmp_path)
    packaged = HarnessTree(candidate)
    shutil.rmtree(tmp_path / "source")
    restored = packaged.materialize("vector",tmp_path / "materialized")
    assert read_json(Path(restored["root"]) / "control.json") == {"coefficient":2.}
    assert verify_harness(restored) == digest(packaged.leaf_identity("vector"))
    source = packaged.root / next(n for n, sha in candidate["components"]["control"].items()
        if n.startswith("objects/") and sha == file_digest(Path(restored["root"]) / "control.json"))
    assert source.stat().st_ino != (Path(restored["root"]) / "control.json").stat().st_ino
    (Path(restored["root"]) / "control.json").write_text("damaged")
    with pytest.raises(ValueError,match="changed"):
        packaged.materialize("vector",tmp_path / "materialized")
    assert HarnessTree(candidate).revision == packaged.revision


def test_branch_components_and_logical_names_cannot_bypass_evolution_limits(tmp_path):
    initial = tree(tmp_path / "initial",nested=False)
    expanded = tree(tmp_path / "expanded")
    assert set(EvolutionSpec(("control","skill_selection")).compare(initial,expanded)) == {"control","skill_selection"}
    with pytest.raises(ValueError,match="fixed evolution"):
        EvolutionSpec(("skill_selection",)).compare(initial,expanded)
    source = leaf(tmp_path / "renamed")
    original = Path(source["root"]) / "foundation.json"
    original.rename(original.with_name("renamed.json"))
    source["components"]["foundation"] = {"renamed.json":source["components"]["foundation"]["foundation.json"]}
    source["file_modes"]["renamed.json"] = source["file_modes"].pop("foundation.json")
    renamed = HarnessTree.create({"fallback":dict(parent=None,when={},harness=source)},keys=["task"],
                                output=tmp_path / "renamed-tree",name="renamed")
    with pytest.raises(ValueError,match="foundation"):
        EvolutionSpec().compare(initial,renamed)
    with pytest.raises(ValueError,match="foundation"):
        HarnessTree.create({"fallback":dict(parent=None,when={},harness=leaf(tmp_path / "normal")),
                           "child":dict(parent="fallback",when={"task":"axis"},harness=source)},
                          keys=["task"],output=tmp_path / "mixed",name="mixed")


@pytest.mark.parametrize("problem", ["overlap","not_narrower","unknown_key","orphan","two_roots","cycle"])
def test_invalid_routes_rejected_before_packaging(tmp_path,problem):
    harness = leaf(tmp_path / "source")
    nodes = {"fallback":dict(parent=None,when={},harness=harness),
             "left":dict(parent="fallback",when={"drive":"x"},harness=harness)}
    if problem == "overlap":
        nodes["right"] = dict(parent="fallback",when={"task":"axis"},harness=harness)
    elif problem == "not_narrower":
        nodes["left"]["when"] = {}
    elif problem == "unknown_key":
        nodes["left"]["when"] = {"private_score":"1"}
    elif problem == "orphan":
        nodes["left"]["parent"] = "missing"
    elif problem == "two_roots":
        nodes["left"].update(parent=None,when={})
    else:
        nodes["right"] = dict(parent="left",when={"drive":"x","task":"axis"},harness=harness)
        nodes["left"]["parent"] = "right"
    with pytest.raises(ValueError):
        HarnessTree.create(nodes,keys=["task","drive"],output=tmp_path / "package",name="invalid")
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("capabilities", [{"task":"axis"}, {"task":"axis","drive":"x","action_contract":"axis","score":"1"},
                                         {"task":"axis","drive":None,"action_contract":"axis"}])
def test_route_requires_exact_public_capabilities(tmp_path,capabilities):
    with pytest.raises(ValueError,match="public capability"):
        HarnessTree(tree(tmp_path)).route(capabilities)


@pytest.mark.parametrize("damage", ["bytes","extra","symlink"])
def test_tree_corruption_is_rejected(tmp_path,damage):
    candidate = tree(tmp_path)
    root = Path(candidate["root"])
    path = root / next(p for p in candidate["components"]["control"] if p.startswith("objects/"))
    if damage == "extra":
        (root / "untracked.json").write_text("{}")
    elif damage == "symlink":
        copy = tmp_path / "same-bytes"
        shutil.copyfile(path,copy)
        path.unlink()
        path.symlink_to(copy)
    else:
        path.write_text("bad")
    with pytest.raises(ValueError):
        HarnessTree(candidate)


def test_routing_pins_one_branch_and_preserves_leaf_context_and_cleanup(tmp_path):
    candidate = tree(tmp_path)
    calls = []

    class ObservedBackend(Backend):
        def build(self,harness,task):
            policy = super().build(harness,task)
            def begin(task,case,observation,context):
                assert context.harness_revision == verify_harness(harness)
                calls.append(("begin",harness["id"]))
            def end(context):
                assert context.harness_revision == verify_harness(harness)
                calls.append(("end",harness["id"]))
                return dict(stopped=True)
            policy.begin_episode, policy.end_episode = begin, end
            return policy

    policy = RoutedPolicy(candidate,task="axis-shifted",backend=ObservedBackend(),extractor=PublicCapabilities(),workspace=tmp_path / "leaves")
    context = Context("routed",harness_revision=verify_harness(candidate))
    observation = dict(position=[0.],target=[2.],drive="shifted",action_contract="axis")
    policy.begin_episode("axis-shifted",{},observation,context)
    assert policy.act(dict(observation,drive="unseen"),context) == 1.
    assert policy.execution_evidence()["path"] == ["fallback","shifted"]
    assert policy.end_episode(context)["stopped"]
    assert calls == [("begin","shifted"),("end","shifted")]
    with pytest.raises(ValueError,match="different candidate"):
        policy.act(observation,replace(context,harness_revision="0"*64))


def test_mismatched_branch_contract_is_rejected_before_execution(tmp_path):
    class BadBackend(Backend):
        def build(self,harness,task):
            policy = super().build(harness,task)
            description = policy.describe()
            if harness["id"] == "shifted":
                policy.describe = lambda: replace(description,action=Contract("wrong-action"))
            return policy
    with pytest.raises(ValueError,match="matching task contracts"):
        RoutedPolicy(tree(tmp_path),task="axis-shifted",backend=BadBackend(),extractor=PublicCapabilities(),workspace=tmp_path / "leaves")


@pytest.mark.parametrize("damage", ["file","cleanup_attribute","inactive_tree_object"])
def test_started_branch_cleanup_survives_runtime_integrity_failure(tmp_path,damage):
    candidate = tree(tmp_path)
    calls=[]
    class WorkerBackend(Backend):
        def build(self,harness,task):
            policy = super().build(harness,task)
            def end(context):
                calls.append((harness["id"],context.harness_revision))
                return dict(stopped=True)
            policy.end_episode=end
            return policy
    suite=Suite(tmp_path / "leaves")
    policy=RoutedPolicy(candidate,task="axis-shifted",backend=WorkerBackend(),extractor=PublicCapabilities(),workspace=suite.workspace)
    expected=digest(HarnessTree(candidate).leaf_identity("shifted"))
    environment=suite.environment("axis-shifted")
    step=environment.step
    def corrupt(action,context):
        result=step(action,context)
        if damage=="file":
            (Path(policy.leaves[policy.selected]["root"])/"control.json").write_text("damaged during execution")
        elif damage=="cleanup_attribute":
            policy.policies[policy.selected].end_episode=None
        else:
            sha=HarnessTree(candidate).leaves["vector"]["control"]["control.json"]
            path = next(n for n, value in candidate["components"]["control"].items() if n.startswith("objects/") and value == sha)
            (Path(candidate["root"])/path).write_text("inactive branch changed")
        return result
    environment.step=corrupt
    runtime=ExperimentRuntime(tmp_path / "runtime")
    with pytest.raises(ValueError):
        runtime.evaluate(Trial("damaged",task="axis-shifted",case=dict(initial=[0.],target=[2.],drive="shifted"),scope="software",budget=Budget(1,30)),
                         environment=environment,policy=policy,verifier=suite.verifier("axis-shifted"))
    assert calls==[("shifted",expected)]
    assert read_json(runtime.root / "damaged/policy.json")["stopped"] is True
    assert read_json(runtime.root / "damaged/receipt.json")["state"]=="needs_reconciliation"


def test_route_evidence_is_owned_and_retained_before_uncertain_action(tmp_path):
    candidate = tree(tmp_path)
    suite = Suite(tmp_path / "leaves")
    environment = suite.environment("axis-shifted")
    def lost(action,context):
        raise RuntimeError("action reply lost")
    environment.step = lost
    policy = suite.policy(candidate,"axis-shifted")
    runtime = ExperimentRuntime(tmp_path / "trials")
    trial = Trial("uncertain",task="axis-shifted",case=dict(initial=[0.],target=[2.],drive="shifted"),scope="software",budget=Budget(1,30))
    for error in (RuntimeError,ReconciliationRequired):
        with pytest.raises(error):
            runtime.evaluate(trial,environment=environment,policy=policy,verifier=suite.verifier("axis-shifted"))
    assert len(suite.resets) == 1
    assert read_json(runtime.root / "uncertain/policy-evidence.json")["path"] == ["fallback","shifted"]
    environment = suite.environment("axis-shifted")
    completed = runtime.evaluate(Trial(**dict(trial.arguments(),run_id="complete")),environment=environment,
                                 policy=suite.policy(candidate,"axis-shifted"),verifier=suite.verifier("axis-shifted"))
    path = runtime.root / "complete/policy-evidence.json"
    assert str(path) in completed.to_dict()["evidence"]
    shutil.rmtree(tmp_path / "leaves")
    assert Evaluation.read(runtime,"complete").revision == completed.revision
    path.write_text("{}")
    with pytest.raises(ValueError,match="evidence changed"):
        runtime.read("complete")


def test_historical_floor_rejects_cumulative_loss_allowed_by_paired_selection(tmp_path):
    # Synthetic scoring fixture for floor arithmetic. The end-to-end study
    # above supplies actual rollout and lineage evidence for integration.
    profile = dict(tasks={task:dict(weight=1,episodes=1,score_range=[0,1],maximum_regression=.2)
                          for task in ("old","new")},minimum_gain=0,tie_tolerance=1e-10)
    state = HarnessState(tmp_path / "state")
    parent = leaf(tmp_path / "source",name="parent")
    initial = state.initialize(parent,policy=dict(profile=profile),scope="software")
    record = {k:v for k,v in initial.items() if k!="revision"}
    historical = dict(parent_id="older",survivor_id="parent",metrics={
        "older":dict(per_task={"old":1.,"new":0.}),"parent":dict(per_task={"old":.9,"new":.4})})
    current = state._publish(dict(record,previous=initial["revision"],action="selection",decision=historical))
    output = tmp_path / "comparison"
    atomic_json(output / "parent.json",current)
    atomic_json(output / "evidence.json",dict(fixture="synthetic scores"))
    comparison = dict(schema_version=1,round_id="next",frozen_at=now(),parent_id="parent",
        candidates={"parent":verify_harness(parent),"child":"2"*64},profile=profile,scope="software")
    cohort = dict(comparison_sha256=digest(comparison),generated_at=now(),layouts={t:[digest(t)] for t in profile["tasks"]})
    results=[]
    for name,scores in (("parent",{"old":.8,"new":.4}),("child",{"old":.75,"new":.8})):
        results.append(dict(candidate_id=name,kind="policy_evaluation",native_exit_code=0,
            comparison_sha256=digest(comparison),cohort_sha256=digest(cohort),freeze_sha256=comparison["candidates"][name],
            evaluator_revision="synthetic",episodes=[dict(task=t,layout_sha256=cohort["layouts"][t][0],state="completed",score=score,
                success=True,evidence_sha256={"evidence.json":file_digest(output / "evidence.json")}) for t,score in scores.items()]))
    assert select_survivor(comparison,cohort,results,evidence_root=output)["survivor_id"] == "child"
    selector = RetentionSelector(state=state,parent_revision=current["revision"],maximum_regression={"old":.2,"new":.2})
    result = selector.select(comparison,cohort,results,evidence_root=output)
    assert result["survivor_id"] == "parent"
    assert result["retention"]["high_water"]["old"] == 1.
    assert result["rejected"]["child"]["tasks"] == ["old"]
    identity = selector.identity()
    state._publish(dict(record,previous=current["revision"],action="selection",decision=historical))
    assert selector.identity() == identity  # A later head cannot alter a frozen comparison.
    with pytest.raises(StateConflict,match="parent changed"):
        state.commit(current["revision"],parent,evidence={},decision=result,selector=selector,comparison_root=output)
    atomic_json(output / "parent.json",initial)
    with pytest.raises(ValueError,match="frozen parent"):
        selector.select(comparison,cohort,results,evidence_root=output)
    (state.root / "history" / (initial["revision"]+".json")).write_text("{}")
    with pytest.raises(ValueError,match="Lineage record changed"):
        selector.identity()


def test_interrupted_tree_publication_never_publishes_partial_package(tmp_path,monkeypatch):
    from PhysicalRSI_core.infra import owned_files
    harness = leaf(tmp_path / "source")
    nodes = {"fallback":dict(parent=None,when={},harness=harness)}
    copying = owned_files.shutil.copyfileobj
    def interrupted(incoming,outgoing,*args,**kwargs):
        copying(incoming,outgoing,*args,**kwargs)
        raise RuntimeError("tree publication interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(owned_files.shutil,"copyfileobj",interrupted)
        with pytest.raises(RuntimeError,match="publication interrupted"):
            HarnessTree.create(nodes,keys=["task"],output=tmp_path / "package",name="tree")
    assert list((tmp_path / "package").iterdir()) == [tmp_path / "package/.lock"]
    candidate = HarnessTree.create(nodes,keys=["task"],output=tmp_path / "package",name="tree")
    assert HarnessTree(candidate).route({"task":"unseen"}) == ["fallback"]


@pytest.mark.parametrize("value", [[], {}, None])
def test_invalid_policy_execution_evidence_cannot_dispatch_an_action(tmp_path,value):
    from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier
    calls=[]
    environment, policy = CounterEnvironment(), CounterPolicy(1)
    policy.execution_evidence = lambda: value
    step = environment.step
    def observed(action,context):
        calls.append(action)
        return step(action,context)
    environment.step = observed
    runtime = ExperimentRuntime(tmp_path / "trials")
    with pytest.raises(ValueError,match="nonempty JSON"):
        runtime.evaluate(Trial("invalid",task="counter",case=dict(initial=0,target=1),scope="software",budget=Budget(1,30)),
                         environment=environment,policy=policy,verifier=CounterVerifier())
    assert not calls
    with pytest.raises(ReconciliationRequired):
        runtime.read("invalid")
