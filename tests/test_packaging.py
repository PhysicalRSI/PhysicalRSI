"""Build and exercise the distributable wheel in an isolated environment."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import textwrap
import venv
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(source, wheelhouse):
    attempts = [
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheelhouse), str(source)],
        [sys.executable, "-m", "pip", "wheel", str(source), "--no-deps",
         "--no-build-isolation", "--wheel-dir", str(wheelhouse)],
    ]
    errors = []
    for command in attempts:
        try:
            subprocess.run(command, check=True, text=True, capture_output=True)
            wheels = sorted(wheelhouse.glob("*.whl"))
            if wheels:
                return wheels[-1]
        except (OSError, subprocess.CalledProcessError) as error:
            output = getattr(error, "stderr", "") or getattr(error, "stdout", "") or str(error)
            errors.append(output[-1000:])
    pytest.skip("wheel build unavailable in this environment: " + " | ".join(errors))


def test_wheel_installs_without_repository_on_sys_path(tmp_path):
    source = tmp_path / "source"

    def ignored(directory, names):
        ignored_names = {".git", ".pytest_cache", "build", "dist"}
        return [name for name in names if name in ignored_names or name.endswith(".egg-info")
                or name == "__pycache__"]

    shutil.copytree(ROOT, source, ignore=ignored)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = _build_wheel(source, wheelhouse)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert not any(name.startswith(("tests/", "examples/")) for name in names)
    assert not any("__pycache__" in name for name in names)
    assert any(name.startswith("PhysicalRSI_core/") for name in names)
    assert any(name.startswith("PhysicalRSI_Autoresearch/") for name in names)
    print(f"wheel={wheel.name} size={wheel.stat().st_size} files={len(names)}")

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--no-deps", str(wheel)],
                   check=True, text=True, capture_output=True)

    smoke = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        import tempfile

        import physicalrsi
        from physicalrsi.core import Budget, ExperimentEvaluator, ImprovementCampaign
        from physicalrsi.core import HarnessState, SelfHarness, TrialQuota
        from PhysicalRSI_core.contracts import Contract
        from PhysicalRSI_core.embodiment import Embodiment, Revision, System1
        from PhysicalRSI_core.infra.storage import atomic_json, file_digest
        from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness

        assert not any("PhysicalRSI-core-dev" in str(path) for path in sys.path)
        assert physicalrsi.__version__ == "0.3.0"

        def manifest(root, name):
            return {"id": name, "root": str(root), "components": {
                kind: {kind + ".json": file_digest(root / (kind + ".json"))}
                for kind in CLOSURE}}

        class Environment:
            def identity(self): return {"kind": "counter-environment", "revision": "1"}
            def describe(self):
                contract = Contract("counter", unit="count", embodiment="software-counter")
                return Embodiment("counter", "1", "software", contract, contract)
            def reset(self, case, context):
                self.value = case["initial"]
                self.target = case["target"]
                return {"ready": True, "observation": {"value": self.value}}
            def step(self, action, context):
                self.value += action
                return {"observation": {"value": self.value},
                        "terminated": self.value >= self.target}

        class Policy:
            def __init__(self, revision): self.revision = revision
            def identity(self): return {"kind": "counter-policy", "revision": "1"}
            def describe(self):
                contract = Contract("counter", unit="count", embodiment="software-counter")
                return System1("counter", self.revision, contract, contract)
            def begin_episode(self, task, case, observation, context): pass
            def act(self, observation, context): return 1

        class Verifier:
            def identity(self): return {"kind": "counter-verifier", "revision": "1"}
            def verify(self, case, trace, context=None):
                return {"outcome": "success" if trace[-1]["observation"]["value"] >= case["target"] else "failure",
                        "reason": "counter", "measurements": {}}

        class Suite:
            def identity(self): return {"kind": "packaging-counter-suite", "revision": "1"}
            def admit(self, candidate):
                return {"accepted": True, "evidence": {"closure": verify_harness(candidate)}}
            def development_cases(self, parent): return {"counter": [{"initial": 0, "target": 2}]}
            def validation_cases(self, comparison): return {"counter": [{"initial": 0, "target": 3}]}
            def environment(self, task): return Environment()
            def policy(self, candidate, task): return Policy(verify_harness(candidate))
            def verifier(self, task): return Verifier()

        class Proposal:
            def __init__(self, evaluator): self.evaluator = evaluator
            def identity(self): return {"kind": "packaging-proposer", "revision": "1"}
            def develop(self, parent, output): return self.evaluator.development(parent, output)
            def propose(self, parent, feedback, output): return []

        class Selector:
            def identity(self): return {"kind": "packaging-selector", "revision": "1"}
            def select(self, *args, **kwargs): raise AssertionError("no candidate should be selected")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed"
            for kind in CLOSURE:
                atomic_json(seed / (kind + ".json"), {"kind": kind})
            initial = manifest(seed, "parent")
            state = HarnessState(root / "state")
            suite = Suite()
            quota = TrialQuota(root / "quota", max_trials=1)
            evaluator = ExperimentEvaluator(
                suite=suite, budgets={"counter": Budget(4, 30)}, scope="software",
                system2=Revision("packaging", "1"), quota=quota)

            def build_loop(directory):
                return SelfHarness(
                    directory, state=state, proposer=Proposal(evaluator), evaluator=evaluator,
                    selector=Selector(), scope="software", max_candidates=1,
                    profile={"tasks": {"counter": {"weight": 1, "episodes": 1,
                        "score_range": [0, 1], "maximum_regression": 0}},
                        "minimum_gain": 0, "tie_tolerance": 1e-10},
                    protocol={"identity": evaluator.revision, "evaluation_kind": "local"})

            campaign = ImprovementCampaign(root / "campaign", state=state,
                                           build_loop=build_loop, max_rounds=1)
            result = campaign.run(initial)
            assert result["status"] == "completed"
            assert result["completed_rounds"] == 1
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run([str(venv_python), "-c", smoke], cwd=tmp_path,
                   env=environment, check=True, text=True)
