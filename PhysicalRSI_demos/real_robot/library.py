"""Immutable reusable code/memory artifacts with PhysicalRSI F/H closure."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

from .capabilities import API
from .contracts import digest, json_copy
from .evidence import write_new
from .models import VisualCodeModel
from .policy_runtime import validate_source
from .references import reference_sources, use_physicalrsi


def dependency_files():
    files = {"adapter/" + p.name: p for p in Path(__file__).parent.glob("*.py")}
    files.update(reference_sources())
    return files


def foundation_identity(value):
    if isinstance(value, dict):
        # Reply routing changes per run, but it does not select another model.
        return {k:foundation_identity(v) for k,v in value.items()
                if not (k == "session" and value.get("kind") == "external-model-mailbox")}
    if isinstance(value, list):
        return [foundation_identity(v) for v in value]
    return value


class PolicyLibrary:
    def __init__(self, root, *, foundation, reusable_skills=()):
        self.reusable_skills = json_copy(list(reusable_skills))
        for skill in self.reusable_skills:
            if (not isinstance(skill, dict) or not isinstance(skill.get("name"), str)
                    or not isinstance(skill.get("source"), str)
                    or skill.get("qualification") != "unvalidated-development-candidate"):
                raise ValueError("Reusable skills require name, source and explicit unvalidated qualification")
            skill["source_sha256"] = hashlib.sha256(skill["source"].encode()).hexdigest()
        self.root = Path(root).resolve()
        self.foundation = foundation_identity(json_copy(foundation))

    def freeze(self, *, task, source, memory):
        validate_source(source)
        if not isinstance(memory, dict):
            raise ValueError("Reusable memory must be an object")
        use_physicalrsi()
        from PhysicalRSI_core.infra.storage import file_digest
        from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
        files = dependency_files()
        dependency_hashes = {name: file_digest(p) for name, p in files.items()}
        identity = digest(dict(task=task.public(), source=source, memory=memory,
                               foundation=self.foundation, api=API, dependencies=dependency_hashes,
                               reusable_skills=self.reusable_skills))
        root = self.root / ("policy-" + identity)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            verify_harness(manifest)
            return manifest
        root.mkdir(parents=True, exist_ok=False)
        (root / "policy.py").write_text(source)
        write_new(root / "memory.json", memory)
        components = {
            "foundation": self.foundation,
            "skill_selection": dict(entry="policy(robot, memory)", level="L1"),
            "skills": dict(entry="policy.py", signature="policy(robot, memory)"),
            "tools": API,
            "control": dict(owner="host RobotPort", judgment_access=False, automatic_rearm=False),
            "prompts": dict(source="adapter/models.py",
                            proposal="reference-source/PhysicalRSI_baselines/robodojo/proposal.py",
                            reusable_skills=self.reusable_skills),
            "memory_rules": dict(source="memory.json", split="development", validation_access=False),
            "dependencies": dependency_hashes,
            "weights": dict(provider_owned=True, trained_here=False),
            "assets": dict(environment_contract="tianji.physical-environment/v1",
                           binding="physical evaluation protocol and per-trial receipts",
                           external_assets="fingerprinted on host; not bundled with reusable policy source"),
            "configuration": dict(task=task.public(), physical_qualification=None),
        }
        assert set(components) == CLOSURE
        for kind, value in components.items():
            write_new(root / (kind + ".json"), value)
        for name, path in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        manifest = dict(id=root.name, root=str(root), components={
            kind: {kind + ".json": file_digest(root / (kind + ".json"))} for kind in components})
        manifest["components"]["skills"]["policy.py"] = file_digest(root / "policy.py")
        manifest["components"]["memory_rules"]["memory.json"] = file_digest(root / "memory.json")
        manifest["components"]["dependencies"].update(dependency_hashes)
        verify_harness(manifest)
        write_new(manifest_path, manifest)
        return manifest

    @staticmethod
    def load(manifest):
        use_physicalrsi()
        from PhysicalRSI_core.self_harness.artifacts import verify_harness
        verify_harness(manifest)
        root = Path(manifest["root"])
        return dict(source=(root / "policy.py").read_text(), memory=json.loads((root / "memory.json").read_text()))

    @staticmethod
    def load_for_execution(manifest, *, foundation=None):
        program = PolicyLibrary.load(manifest)
        from PhysicalRSI_core.infra.storage import file_digest
        root = Path(manifest["root"])
        frozen = json.loads((root / "dependencies.json").read_text())
        saved_foundation = json.loads((root / "foundation.json").read_text())
        if saved_foundation != foundation_identity(saved_foundation):
            raise ValueError("Legacy foundation includes reply routing; refreeze as a development candidate")
        current = {name:file_digest(p) for name, p in dependency_files().items()}
        if frozen != current or json.loads((root / "tools.json").read_text()) != API:
            raise ValueError("Policy adapter/API changed; refreeze as a development candidate and reevaluate")
        if foundation is not None:
            if saved_foundation != foundation_identity(foundation):
                raise ValueError("Policy foundation changed; independent comparison does not transfer")
        return program

    def propose(self, *, task, parent, feedback, model, output, seed=None):
        from .learning_projection import proposal_feedback
        use_physicalrsi()
        from PhysicalRSI_baselines.robodojo.proposal import CodeProposer
        from PhysicalRSI_baselines.robodojo.skill_proposal import propose
        output = Path(output)
        previous = self.load(parent) if parent else (seed or dict(source="", memory={}))
        skills = json_copy(self.reusable_skills)
        proposal = propose(task=task.id, source=previous["source"], memory=previous["memory"],
                           feedback=proposal_feedback(feedback), skill_api=API,
                           proposer=CodeProposer(VisualCodeModel(model, output=output, reusable_skills=skills)), output=output)
        tree = ast.parse(proposal["source"])
        write_new(output / "reusable-skill-use.json", {skill["name"]: dict(
            source_supplied=True, supplied_sha256=skill["source_sha256"],
            defined=any(isinstance(n, ast.FunctionDef) and n.name == skill["name"] for n in ast.walk(tree)),
            referenced_by_call=any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                   and n.func.id == skill["name"] for n in ast.walk(tree)),
            executed=False, physically_validated=False) for skill in skills})
        if parent and proposal["source"] == previous["source"] and proposal["memory"] == previous["memory"]:
            return None
        return self.freeze(task=task, source=proposal["source"], memory=proposal["memory"])
