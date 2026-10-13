"""Two real CPU REINFORCE updates through the optional checkpoint evolver.

The finite categorical policy learns from observed rollout rewards. No simulator,
GPU trainer, external framework, model weights or physical qualification implied.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from PhysicalRSI_core.checkpoints import CheckpointStore
from PhysicalRSI_core.contracts import Context, Contract, Operation, ReconciliationRequired
from PhysicalRSI_core.embodiment import Dependency, Revision, System1
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.scheduler import JobScheduler
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.training import (CheckpointEvolutionSpec, CheckpointProposer,
                                                   TrainingCoordinator, TrainingPending)
from .recursive_research import Selector, manifest
from .rollout_evolution import Suite as CalibrationSuite, contracts

SCALES = (1., .5, 2.)
TASKS = ("axis", "vector")


def identity(kind, **values):
    return dict(kind=kind, implementation=file_digest(Path(__file__)), **values)


class Backend:
    def identity(self):
        return identity("categorical-linear-inference")

    def build(self, root, entrypoint, model):
        if model != model_spec():
            raise ValueError("Backend requires its declared model architecture and interface")
        weights = read_json(root / entrypoint)
        if (set(weights) != {"logits"} or len(weights["logits"]) != 2
                or any(len(row) != 3 or any(type(x) not in {int, float} or not math.isfinite(x) for x in row)
                       for row in weights["logits"])):
            raise ValueError("Expected a finite two-feature, three-action checkpoint")
        # Own these loaded values; inference never reads a mutable latest path.
        rows = deepcopy(weights["logits"])
        def predict(feature, context):
            if type(feature) is not int or feature not in {0, 1}:
                raise ValueError("Unknown model feature")
            values = [math.exp(x - max(rows[feature])) for x in rows[feature]]
            return [x / sum(values) for x in values]
        return Operation(model["name"], digest(weights), Contract(**model["input"]),
                         Contract(**model["output"]), predict)


def model_spec():
    return dict(name="calibration-model", format="json-categorical-linear/v1",
        architecture=dict(features=list(TASKS), scales=list(SCALES)), backend=Backend().identity(),
        input=asdict(Contract("calibration-feature")), output=asdict(Contract("categorical-probabilities")), effects=[])


class Policy:
    def __init__(self, task, candidate, operation):
        self.task, self.candidate, self.operation = task, verify_harness(candidate), operation

    def identity(self):
        return identity("checkpoint-calibration-policy", task=self.task, candidate=self.candidate,
                        checkpoint=self.operation.revision)

    def describe(self):
        return System1("calibration", self.candidate, *contracts(self.task),
                       dependencies=(Dependency(self.operation.name, self.operation.revision, "model"),))

    def begin_episode(self, task, case, observation, context):
        self.probabilities = self.operation(TASKS.index(task), context)
        self.choice = max(range(3), key=self.probabilities.__getitem__)
        if "draw" in case:
            cumulative = 0.
            for index, probability in enumerate(self.probabilities):
                cumulative += probability
                if case["draw"] < cumulative:
                    self.choice = index
                    break
        self.sampled = "draw" in case

    def act(self, observation, context):
        commands = [SCALES[self.choice] * (t - p) for p,t in zip(observation["position"], observation["target"])]
        return commands[0] if self.task == "axis" else dict(joint_delta=commands)

    def execution_evidence(self):
        return dict(schema="categorical-action/v1", feature=TASKS.index(self.task), probabilities=self.probabilities,
                    choice=self.choice, sampled=self.sampled, checkpoint=self.operation.revision)


class Suite(CalibrationSuite):
    def __init__(self, store):
        super().__init__({"axis":2., "vector":.5})
        self.store = store

    def identity(self):
        return identity("trained-calibration-suite", calibration=super().identity(), store=self.store.identity())

    def admit(self, candidate):
        self.store.binding(candidate, model=model_spec())
        return super().admit(candidate)

    def policy(self, candidate, task):
        return Policy(task, candidate, self.store.bind(candidate, backend=Backend(), model=model_spec()))


def samples(values):
    result = []
    for value in values:
        paths = [p for p in value["evidence"] if p.endswith("/policy-evidence.json")]
        if len(paths) != 1:
            raise ValueError("Require actual model action evidence")
        evidence = read_json(paths[0])
        if evidence["schema"] != "categorical-action/v1" or evidence["sampled"] is not True:
            raise ValueError("Training needs sampled development behavior")
        result.append((value, evidence))
    return result


class Planner:
    def identity(self):
        return identity("first-unsolved-feature-plan")

    def plan(self, parent, values, context, output):
        rows = samples(values)
        for task in TASKS:
            group = [(v,e) for v,e in rows if v["trial"]["task"] == task]
            greedy = max(range(3), key=group[0][1]["probabilities"].__getitem__)
            if not any(v["outcome"] == "success" and e["choice"] == greedy for v,e in group):
                if any(v["outcome"] == "success" for v,e in group):
                    return dict(parameters=dict(task=task, learning_rate=8.), used_lessons=[],
                        hypothesis="Increase probability of rewarded actions for " + task,
                        expected_effect="Improve held-out targets while preserving the other feature",
                        required_checks=["paired-position-success", "per-task-nonregression"])
        return None

    resume_plan = plan  # This local planner is a pure read-only calculation.


class TrainingHandler:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = CheckpointStore(self.root / "models")

    def identity(self):
        return identity("cpu-reinforce-worker", root=str(self.root), store=self.store.identity())

    def execute(self, request):
        started = monotonic()
        key = digest(request)
        directory = self.root / "training-jobs" / key
        if (directory / "result.json").exists():
            raise ReconciliationRequired("Collect the existing training result; do not execute twice")
        values = self.store.training_source(request)
        parent = self.store.read(request["parent_checkpoint"])
        root = self.store.checkpoints.folder(request["parent_checkpoint"]) / "files"
        weights = read_json(root / parent["metadata"]["entrypoint"])
        operation = Backend().build(root, parent["metadata"]["entrypoint"], request["model"])
        if len(values) > request["budget"]["max_samples"] or request["budget"]["max_updates"] < 1:
            raise ValueError("Training data exceeds its declared budget")
        selected, rewards = [], []
        for value, evidence in samples(values):
            feature = TASKS.index(value["trial"]["task"])
            probabilities = operation(feature, Context("training-check"))
            if (evidence["feature"] != feature or evidence["checkpoint"] != request["parent_checkpoint"]
                    or evidence["probabilities"] != probabilities):
                raise ValueError("Training behavior differs from the on-policy parent")
            choice = evidence["choice"]
            before, after = value["trajectory"]
            commands = [SCALES[choice] * (t-p) for p,t in zip(before["observation"]["position"],
                                                           before["observation"]["target"])]
            if after["action"] != (commands[0] if feature == 0 else dict(joint_delta=commands)):
                raise ValueError("Recorded model choice differs from actual actions")
            reward = float(value["outcome"] == "success")
            if value["trial"]["task"] == request["parameters"]["task"]:
                selected.append((probabilities, choice, reward))
                rewards.append(reward)
        if not selected:
            raise ValueError("No on-policy samples for the chosen feature")
        feature = TASKS.index(request["parameters"]["task"])
        learning_rate = request["parameters"]["learning_rate"]
        # One actual REINFORCE gradient step; no environment gains are read.
        for j in range(3):
            gradient = sum(reward * (float(choice == j) - p[j]) for p,choice,reward in selected) / len(selected)
            weights["logits"][feature][j] += learning_rate * gradient
        elapsed = monotonic() - started
        if elapsed > request["budget"]["max_seconds"]:
            raise ValueError("Training exceeded its time budget")
        bundle = directory / "bundle"
        atomic_json(bundle / "weights.json", weights)
        result = dict(root=str(bundle), entrypoint="weights.json", files={"weights.json":file_digest(bundle / "weights.json")},
            costs=dict(updates=1, samples=len(values), seconds=elapsed), metrics=dict(mean_sample_reward=sum(rewards)/len(rewards)),
            role="training_only", request_sha256=key, job_id=job_id(request, self))
        atomic_json(directory / "result.json", result)
        return self.collect(request)

    def collect(self, request):
        result = read_json(self.root / "training-jobs" / digest(request) / "result.json")
        if result["request_sha256"] != digest(request) or result["job_id"] != job_id(request, self):
            raise ValueError("Training result changed attribution")
        for name, sha in result["files"].items():
            if file_digest(Path(result["root"]) / name) != sha:
                raise ValueError("Training result bytes changed")
        return result


def job_id(request, handler):
    return digest(dict(payload=request, handler=handler.identity(), resources={"cpu":1}, priority=0))


class Provider:
    def __init__(self, root):
        self.handler = TrainingHandler(root)
        self.scheduler = JobScheduler(Path(root) / "training-queue", capacities={"cpu":1})

    def identity(self):
        return identity("local-training-provider", handler=self.handler.identity(), scheduler=self.scheduler.identity())

    def submit(self, request, output):
        key, = self.scheduler.submit([request], handler=self.handler, resources={"cpu":1})
        return dict(job_id=key, request_sha256=digest(request))

    def recover_submission(self, request, output):
        key = job_id(request, self.handler)
        job = self.scheduler.status()["jobs"].get(key)
        if job is None or job["request"]["payload"] != request:
            return None
        return dict(job_id=key, request_sha256=digest(request))

    def poll(self, handle):
        job = self.scheduler.status()["jobs"].get(handle["job_id"])
        state = "unknown" if job is None else job["state"]
        if job is not None and (digest(job["request"]["payload"]) != handle["request_sha256"]
                or state == "needs_reconciliation" or state == "running" and not job["owner_active"]):
            state = "unknown"
        if state == "abandoned":
            state = "failed"  # Explicit operator evidence established quiescence.
        return dict(handle, state=state, quiescent=state in {"completed", "failed"})

    def collect(self, handle):
        job = self.scheduler.status([handle["job_id"]])["jobs"][handle["job_id"]]
        if job["state"] != "completed":
            raise ReconciliationRequired("Training job is not known to be complete")
        result = self.handler.collect(job["request"]["payload"])
        if result != job["result"]:
            raise ValueError("Completed queue result changed")
        return result


class Reflector:
    def identity(self):
        return identity("checkpoint-no-memory-update")

    def reflect(self, context, development, output):
        return dict(schema="physicalrsi.memory-delta/v1", operations=[])


def setup(root):
    root = Path(root).resolve()
    store = CheckpointStore(root / "models")
    provider = Provider(root)
    backend, model = Backend(), model_spec()
    raw = root / "seed-input"
    if not raw.exists():
        for kind in CLOSURE:
            atomic_json(raw / (kind + ".json"), dict(kind=kind))
        atomic_json(root / "seed-model/weights.json", dict(logits=[[1.,0.,0.], [1.,0.,0.]]))
    seed = store.publish(model=model, backend=backend, root=root / "seed-model", entrypoint="weights.json",
                         files={"weights.json":file_digest(root / "seed-model/weights.json")})
    initial = store.install(manifest(raw, "untrained-input"), seed, root / "seed", candidate_id="untrained")
    state, suite = HarnessState(root / "state"), Suite(store)
    quota = TrialQuota(root / "quota", max_trials=52)
    memory = ResearchMemory(root / "memory")
    memory_revision = memory.write({"baseline":dict(conditions={"study":"checkpoint-training"},
        hypothesis="Learn model actions from development rewards", required_checks=["paired-position-success"],
        counterexample_checks=[], evidence={str(root / "seed-model/weights.json"):file_digest(root / "seed-model/weights.json")})})
    coordinator = TrainingCoordinator(store=store, provider=provider, backend=backend)
    def build(directory, context):
        def cases(targets, explore=False):
            return {task:[dict(initial=[0] * (1 if task == "axis" else 2),
                              target=[n] if task == "axis" else [n,-2*n],
                              **({"draw":(i+.5)/8} if explore else {})) for i,n in enumerate(targets)] for task in TASKS}
        protocol = CaseProtocol(directory / "cases", splits=dict(
            development=cases(range(1+20*context["index"],9+20*context["index"]), True),
            validation=cases([101+4*context["index"],103+4*context["index"]]), test=cases([201,203])),
            identity_keys=("initial","target"))
        evaluator = ExperimentEvaluator(suite=PreregisteredSuite(suite, protocol),
            budgets={task:Budget(1,30) for task in TASKS}, scope="software",
            system2=Revision("cpu-reinforce", "1"), quota=quota)
        proposer = CheckpointProposer(evaluator, coordinator, Planner(), model=model,
                                     budget=dict(max_updates=1,max_samples=16,max_seconds=300))
        return SelfHarness(directory, state=state, proposer=proposer, evaluator=evaluator,
            selector=Selector(), evolution=CheckpointEvolutionSpec(store,model), max_candidates=1,
            scope="software", round_context=context,
            profile=dict(tasks={task:dict(weight=1,episodes=2,score_range=[0,1],maximum_regression=0) for task in TASKS},
                         minimum_gain=0, tie_tolerance=1e-10),
            protocol=dict(identity=evaluator.revision,evaluation_kind="local_contract_evaluation"))
    campaign = ResearchCampaign(root / "campaign",state=state,build_loop=build,memory=memory,
        memory_revision=memory_revision,context={"study":"checkpoint-training"},reflector=Reflector(),max_rounds=2,
        update_memory=False)
    return campaign, initial, suite, quota, provider


def worker(root, stop):
    provider = Provider(root)
    while not stop.exists():
        if provider.scheduler.work_once(provider.handler) is None:
            sleep(.05)


def run(root):
    root = Path(root).resolve()
    campaign, initial, suite, quota, provider = setup(root)
    process, log = None, None
    stop = root / "worker-stop"
    deadline = monotonic() + 900
    try:
        while True:
            try:
                research = campaign.run(initial)
                break
            except TrainingPending:
                if monotonic() > deadline:
                    raise ReconciliationRequired("Observe the pending job before continuing; training was not resubmitted")
                if process is None:
                    stop.unlink(missing_ok=True)
                    log = (root / "worker.log").open("a")
                    process = subprocess.Popen([sys.executable,"-m","PhysicalRSI_demos.checkpoint_evolution","--worker",str(root)],stdout=log,stderr=log)
                    atomic_json(root / "worker-process.json",dict(pid=process.pid, provider=provider.identity(),
                        command=[sys.executable,"-m","PhysicalRSI_demos.checkpoint_evolution","--worker",str(root)]))
                elif process.poll() is not None:
                    raise ReconciliationRequired("Training worker exited; reconcile its original job")
                sleep(.05)
        return dict(research=research, final_test=campaign.report_test(initial), quota=quota.status(),
                    training_jobs=provider.scheduler.status())
    finally:
        if process is not None:
            stop.write_text("stop after current job\n")
            # A timeout does not terminate an uncertain trainer or release its
            # reservation. The caller must reconcile the recorded worker/job.
            process.wait(timeout=30)
            log.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[2]) / "worker-stop")
    elif len(sys.argv) == 2:
        import json
        print(json.dumps(run(sys.argv[1]), indent=2, sort_keys=True))
    else:
        raise SystemExit("python -m PhysicalRSI_demos.checkpoint_evolution WORKSPACE")
