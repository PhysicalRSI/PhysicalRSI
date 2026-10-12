"""Actual PhysicalRSI propose/evaluate/select/inherit, using independent episodes."""
from datetime import datetime, timezone
from pathlib import Path
import json

from .contracts import digest, json_copy
from .evidence import write_new
from .library import PolicyLibrary, foundation_identity
from .references import use_physicalrsi

use_physicalrsi()
from PhysicalRSI_core.infra.storage import digest as rsi_digest, file_digest, atomic_json, locked
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.lineage import HarnessState


class DevelopmentProposer:
    def __init__(self, *, library, task, model, feedback):
        if not feedback or any(r.get("split") != "development" or r.get("task") != task.id for r in feedback):
            raise ValueError("Only task-matched development demonstrations may improve L1")
        self.library, self.task, self.model = library, task, model
        self.feedback = json_copy(feedback)
        self.feedback_sha256 = digest(self.feedback)

    def identity(self):
        return dict(kind="PhysicalRSI-CodeProposer-with-L2-L3-demonstrations",
                    foundation=foundation_identity(self.model.config.public()),
                    task=self.task.public(),
                    implementation=file_digest(Path(__file__)))

    def bind_round(self, parent, root):
        """Bind changing development inputs outside the stable lineage policy."""
        root = Path(root).resolve()
        expected = dict(schema="tianji.evolution-input/v1", task=self.task.public(),
                        parent_sha256=verify_harness(parent), feedback_sha256=self.feedback_sha256)
        with locked(root / ".development-input.lock"):
            path = root / "development-input.json"
            if path.exists():
                if json.loads(path.read_text()) != expected:
                    raise ValueError("This round is already bound to different development input or parent")
            else:
                # Never retrofit a binding over a previously started experiment.
                if (root / "parent.json").exists() or (root / "develop.json").exists():
                    raise ValueError("Legacy started round has no input binding; preserve it and use a new round")
                atomic_json(path, expected)
        self.assert_round(root)

    def assert_round(self, root):
        root = Path(root)
        binding = json.loads((root / "development-input.json").read_text())
        if (digest(self.feedback) != self.feedback_sha256
                or binding.get("feedback_sha256") != self.feedback_sha256
                or binding.get("task") != self.task.public()):
            raise ValueError("Bound development input changed")
        anchor = root / "parent.json"
        if anchor.exists() and json.loads(anchor.read_text())["freeze_sha256"] != binding["parent_sha256"]:
            raise ValueError("Bound parent differs from the round's committed anchor")
        public = root / "development-public.json"
        if public.exists() and digest(json.loads(public.read_text())) != self.feedback_sha256:
            raise ValueError("Saved development feedback changed")

    def develop(self, parent, output):
        path = output / "development-public.json"
        write_new(path, self.feedback)
        return dict(split="evolve", evidence={str(path): file_digest(path)})

    def propose(self, parent, feedback, output):
        self.assert_round(output)
        for path, sha in feedback["evidence"].items():
            if file_digest(Path(path)) != sha:
                raise ValueError("Development demonstration changed")
        child = self.library.propose(task=self.task, parent=parent, feedback=self.feedback,
                                     model=self.model, output=output / "proposal")
        if child is None:
            return []
        child.update(parent_sha256=verify_harness(parent), method="PhysicalRSI CodeProposer",
                     changes="Reusable source and memory learned from public L2/L3 development trajectories",
                     environment="Tianji dual-system host adapter; no physical qualification implied",
                     evidence=feedback["evidence"], costs=dict(model_proposals=1))
        return [child]


class PairedEpisodeEvaluator:
    """Caller supplies held-out scenes and runs L1-only independently judged trials.

    scene_provider is called only AFTER candidate-pool freeze. It owns physical
    scene restoration and any privileged reference; neither reaches the proposer.
    trial_runner must run the frozen candidate, disable L2/L3 rescue, and save all
    camera/command/judgment evidence in the supplied directory.
    """
    def __init__(self, *, task, scene_provider, trial_runner, identity, episodes, qualification="offline", environment=None, environment_capture=None):
        if type(episodes) is not int or episodes <= 0:
            raise ValueError("Positive paired validation count required")
        if qualification not in {"offline", "physical"}:
            raise ValueError("Declare offline or physical evaluation qualification")
        if qualification == "physical" and (not isinstance(environment, dict)
                or environment.get("schema") != "tianji.physical-environment/v1"):
            raise ValueError("Physical comparison requires a declared software/configuration/asset environment")
        if qualification == "physical" and not callable(environment_capture):
            raise ValueError("Physical comparison requires an explicit host environment capture")
        self.environment_capture = environment_capture
        self.task, self.scene_provider, self.trial_runner = task, scene_provider, trial_runner
        self.declared_identity, self.episodes = identity, episodes
        self.qualification = qualification
        self.environment = json_copy(environment)
        self.revision = rsi_digest(self.identity())

    def identity(self):
        return dict(kind="independent-L1-only-paired-episodes", task=self.task.public(),
                    adapter=self.declared_identity, episodes=self.episodes,
                    qualification=self.qualification,
                    environment=self.environment,
                    implementation=file_digest(Path(__file__)))

    def admit(self, candidate, output):
        PolicyLibrary.load_for_execution(candidate)
        path = output / ("admission-" + candidate["id"] + ".json")
        receipt = dict(accepted=True, freeze_sha256=verify_harness(candidate),
                       scope="source/closure only; scene transfer is tested by validation", qualification=None)
        write_new(path, receipt)
        return dict(receipt, evidence={str(path): file_digest(path)})

    def validation(self, comparison, output):
        cases = self.scene_provider(self.task, self.episodes)
        if not isinstance(cases, list) or len(cases) != self.episodes or any(not isinstance(c, dict) for c in cases):
            raise ValueError("Held-out scene inventory differs")
        identities = [rsi_digest(case) for case in cases]
        if len(set(identities)) != len(identities):
            raise ValueError("Validation scenes must be distinct")
        path = output / "validation-scenes.json"
        write_new(path, dict(zip(identities, cases)))
        return dict(split="validation", comparison_sha256=rsi_digest(comparison),
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    layouts={self.task.id: identities},
                    admission_evidence={str(path): file_digest(path)})

    def evaluate(self, candidate, comparison, cohort, output):
        path = output / "validation-scenes.json"
        if cohort["admission_evidence"].get(str(path)) != file_digest(path):
            raise ValueError("Validation scene inventory changed")
        cases = json.loads(path.read_text())
        rows = []
        for case_id in cohort["layouts"][self.task.id]:
            directory = output / "trials" / candidate["id"] / case_id
            # The runner owns creation, durable receipts and physical handoff.
            # A started unknown trial is never automatically replayed on resume.
            result = self.trial_runner(self.task, cases[case_id], candidate, directory)
            if (result.get("split") != "validation" or result.get("policy_id") != candidate["id"]
                    or result.get("status") != "completed" or result.get("outcome") not in {"success", "failure"}
                    or result.get("l2_calls") != 0 or result.get("l3_calls") != 0
                    or result.get("qualification") != self.qualification or result.get("task") != self.task.id):
                raise ValueError("Validation needs definite independent L1-only results with matching qualification/task")
            success = result["outcome"] == "success"
            if result.get("l1_success") is not success or result.get("solved_by") != ("L1" if success else None):
                raise ValueError("Validation outcome and L1 attribution disagree")
            if json.loads((directory / "result.json").read_text()) != result:
                raise ValueError("Trial return differs from durable result evidence")
            metadata = json.loads((directory / "episode.json").read_text())
            if (metadata.get("task") != self.task.public() or metadata.get("policy_id") != candidate["id"]
                    or metadata.get("split") != "validation" or metadata.get("qualification") != self.qualification):
                raise ValueError("Trial metadata differs from the declared paired evaluation")
            if self.qualification == "physical":
                from .host_environment import require_same_environment, verify_trial_environment
                require_same_environment(self.environment, self.environment_capture())
                verify_trial_environment(directory, self.environment)
            files = {}
            for p in directory.rglob("*"):
                if p.is_symlink():
                    raise ValueError("Validation evidence must be physical files")
                if p.is_file():
                    files[str(p.relative_to(output))] = file_digest(p)
            if not files:
                raise ValueError("Validation evidence is empty")
            rows.append(dict(task=self.task.id, layout_sha256=case_id, state="completed",
                             success=success, score=float(success), evidence_sha256=files))
        return dict(kind="tianji_l1_policy_evaluation", native_exit_code=0,
                    candidate_id=candidate["id"], freeze_sha256=verify_harness(candidate),
                    comparison_sha256=rsi_digest(comparison), cohort_sha256=rsi_digest(cohort),
                    evaluator_revision=self.revision, episodes=rows)


class Selector:
    def identity(self):
        return dict(implementation=file_digest(Path(selection.__file__)))

    select = staticmethod(selection.select_survivor)


class BoundDevelopmentHarness(SelfHarness):
    """Original SelfHarness phases plus a per-round development input check."""
    def _stable(self):
        super()._stable()
        self.proposer.assert_round(self.root)

    def _run(self):
        # The upstream completed-commit fast path precedes its first _step.
        # Validate here too, so resuming a completed round cannot change inputs.
        self._stable()
        return super()._run()


def make_harness(root, *, state_root, parent, proposer, evaluator):
    state = HarnessState(state_root)
    root = Path(root).resolve()
    verify_harness(parent)
    parent_task = json.loads((Path(parent["root"]) / "configuration.json").read_text())["task"]
    if proposer.task.public() != evaluator.task.public() or parent_task != evaluator.task.public():
        raise ValueError("Parent, proposer and evaluator must use the same task and success criterion")
    if not (root / "parent.json").exists() and (state.root / "current.json").exists():
        if state.resolve()["freeze_sha256"] != verify_harness(parent):
            raise ValueError("New round must use the current lineage parent")
    proposer.bind_round(parent, root)
    profile = dict(tasks={evaluator.task.id: dict(weight=1, episodes=evaluator.episodes,
                  score_range=[0, 1], maximum_regression=0)}, minimum_gain=0, tie_tolerance=0)
    loop = BoundDevelopmentHarness(root, state=state, proposer=proposer, evaluator=evaluator,
                       selector=Selector(), profile=profile, max_candidates=1,
                       protocol=dict(identity=evaluator.revision, evaluation_kind="tianji_l1_policy_evaluation"),
                       scope="Independent paired L1-only success; no automatic physical qualification")
    state.initialize(parent, policy=loop.config, scope=loop.config["scope"])
    return loop
