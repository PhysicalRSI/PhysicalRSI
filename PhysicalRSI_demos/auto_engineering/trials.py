"""Native Isaac adapters for the common Trial/Evaluation execution path.

One runtime action dispatches a complete public pick/place plan. The fixed
Isaac controller executes it at 60 Hz; its actual joint targets and observations
are retained separately in public-rollout.json. There is no surrogate backend.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from threading import local

from PhysicalRSI_core.contracts import Contract, ReconciliationRequired
from PhysicalRSI_core.embodiment import Embodiment, System1
from PhysicalRSI_core.experiments import TrialAdapters
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from .harness import implementation, verify_episode
from .lab_style import asset_identity
from .policy import plan
from .task import TASK, SCOPE, judge, layout


def contracts():
    return (Contract("lab-public-rgbd-and-joints/v1", embodiment="isaac-franka"),
            Contract("lab-pick-place-plan/v1", unit="metre", frame="world", embodiment="isaac-franka"))


class NativeWorker:
    def __init__(self, python, root, *, gui=False, cache_root=None):
        # Preserve a virtual environment's launcher path: resolving its symlink
        # would execute the base interpreter without the installed Isaac packages.
        self.python, self.root, self.gui = str(Path(python).expanduser().absolute()), Path(root).resolve(), gui
        self.cache_root = Path(cache_root).resolve() if cache_root is not None else None
        self.process = None

    def identity(self):
        executable = Path(self.python)
        return dict(kind="isaac51-native-worker/v1", python=self.python,
                    executable_sha256=file_digest(executable), gui=self.gui,
                    cache_root=str(self.cache_root) if self.cache_root is not None else None,
                    implementation=implementation(), assets=asset_identity())

    def _prepare_cache(self):
        """Link only derived Kit cache data; never merge existing runtime trees."""
        if self.cache_root is None:
            return self.root / "runtime"
        target = (self.cache_root / "kit").resolve()
        local_cache = self.root / "runtime/kit/cache"
        if target.is_relative_to(local_cache) or local_cache.is_relative_to(target):
            raise ValueError("Shared cache must be outside the local Kit cache")
        if local_cache.is_symlink():
            if local_cache.resolve() != target:
                raise ValueError("Existing Kit cache points to a different shared cache")
        elif local_cache.exists():
            raise ValueError("Existing local Kit cache cannot be replaced by a shared cache")
        target.mkdir(parents=True, exist_ok=True)
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        if not local_cache.is_symlink():
            local_cache.symlink_to(target, target_is_directory=True)
        return self.cache_root

    def _wait(self, context, *, path=None):
        while True:
            context.check()
            code = self.process.poll()
            if path is not None and path.exists():
                return read_json(path)
            if code is not None:
                if code:
                    raise RuntimeError("Native Isaac worker failed; inspect " + str(self.directory / "worker.log"))
                if path is not None:
                    raise RuntimeError("Native worker exited before reset readiness")
                return
            time.sleep(.05)

    def reset(self, case, context):
        if set(case) != {"seed"} or type(case["seed"]) is not int:
            raise ValueError("Native case requires one integer seed")
        self.directory = self.root / (context.episode + "-" + context.harness_revision)
        if self.directory.exists():
            raise ReconciliationRequired("Native output already exists; reconcile before retry: " + str(self.directory))
        self.directory.mkdir(parents=True)
        self.ipc, self.output = self.directory / "ipc", self.directory / "episode"
        self.ipc.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMNI_KIT_ACCEPT_EULA="YES",
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        if os.geteuid() == 0:
            env["OMNI_KIT_ALLOW_ROOT"] = "1"
        cache = self._prepare_cache()
        for key, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                          ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                          ("CUDA_CACHE_PATH", "cuda-cache"), ("__GL_SHADER_DISK_CACHE_PATH", "gl-cache")):
            base = cache if key in {"XDG_CACHE_HOME", "CUDA_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH"} else self.root / "runtime"
            path = base / name
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        icd = self.root / "runtime" / "nvidia-vulkan.json"
        atomic_json(icd, dict(file_format_version="1.0.0", ICD=dict(
            library_path="libGLX_nvidia.so.0" if self.gui else "libEGL_nvidia.so.0", api_version="1.3.194")))
        env["VK_ICD_FILENAMES"] = str(icd)
        command = [self.python, "-m", "PhysicalRSI_demos.auto_engineering.isaac_scene",
                   "--seed", str(case["seed"]), "--bridge", str(self.ipc), "--output", str(self.output),
                   "--runtime-root", str(self.root / "runtime" / "kit")]
        if self.gui:
            command.append("--gui")
        try:
            with (self.directory / "worker.log").open("x") as log:
                self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return self._wait(context, path=self.ipc / "ready.json")
        except BaseException:
            self.close()
            raise

    def execute(self, action, context):
        try:
            atomic_json(self.ipc / "action.json", action)
            self._wait(context)
            result = verify_episode(self.output)
            rollout = read_json(self.output / "public-rollout.json")
            return result, rollout
        finally:
            self.close()

    def close(self):
        if self.process is None:
            return
        # Stop this owned group even if its leader already exited; a child must
        # not outlive the trial and keep using its leased simulator resources.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        self.process = None

    def exports(self, expected):
        payload = (self.output / "result.json").read_bytes()
        if json.loads(payload) != expected:
            raise ValueError("Native result changed before evidence capture")
        files = dict(expected["evidence"], **{"result.json": sha256(payload).hexdigest()})
        return {"native/" + name: dict(path=str(self.output / name), sha256=sha) for name, sha in files.items()}


class Environment:
    def __init__(self, worker, resource):
        self.worker, self.resource = worker, resource
        self.result = None

    def identity(self):
        return dict(kind="isaac-trial-environment/v1", worker=self.worker.identity(), resource=self.resource)

    def describe(self):
        return Embodiment("isaac-lab", digest(self.identity()), "simulation", *contracts(), resources=(self.resource,))

    def reset(self, case, context):
        self.case = deepcopy(case)
        self.initial = self.worker.reset(case, context)
        return deepcopy(self.initial)

    def step(self, action, context):
        self.result, rollout = self.worker.execute(action, context)
        if (self.result["seed"] != self.case["seed"] or self.result["memory"] != action["memory"]
                or self.result["layout_sha256"] != digest(layout(self.case["seed"]))
                or self.result.get("runtime_version") not in {"5.1.0", "5.1.0.0"}
                or self.result["scope"] != SCOPE or self.result["qualification"] is not None):
            raise ValueError("Native episode attribution differs from the Trial")
        if (not rollout or len(rollout) != self.result["physics_steps"] + 1
                or rollout[0]["observation"] != self.initial["observation"]
                or any(row.get("step") != i or "action" not in row for i, row in enumerate(rollout[1:]))):
            raise ValueError("Native public rollout is incomplete or belongs to another reset")
        return dict(terminated=True, observation=deepcopy(rollout[-1]["observation"]),
                    native_rollout=rollout)

    def quiesce(self, context):
        self.worker.close()
        return dict(quiescent=True, reason="Owned native worker exited")

    def export_evidence(self):
        return self.worker.exports(self.result) if self.result is not None else {}


class Policy:
    def __init__(self, candidate, environment):
        self.candidate, self.environment = deepcopy(candidate), environment

    def identity(self):
        return dict(kind="public-lab-plan/v1", candidate=verify_harness(self.candidate), implementation=implementation())

    def describe(self):
        return System1("isaac-lab-code-policy", verify_harness(self.candidate), *contracts())

    def begin_episode(self, task, case, observation, context):
        if task != TASK:
            raise ValueError("Unknown Isaac task")

    def act(self, observation, context):
        memory = read_json(Path(self.candidate["root"]) / "memory_rules.json")
        return dict(memory=memory, plan=plan(observation["public_observation"], memory))

    def end_episode(self, context):
        self.environment.worker.close()
        return dict(stopped=True)


class Verifier:
    def __init__(self, environment):
        self.environment = environment

    def identity(self):
        return dict(kind="independent-native-lab-judge/v1", implementation=implementation())

    def verify(self, case, trace):
        worker = self.environment.worker
        result = verify_episode(worker.output)
        measured = read_json(worker.output / "private-evaluation.json")
        verdict = judge(layout(case["seed"]), measured)
        if not result["controller_completed"]:
            verdict["success"] = False
            verdict["checks"]["controller_completed"] = False
        if any(result[key] != verdict[key] for key in ("success", "checks", "qualification", "scope")):
            raise ValueError("Worker outcome disagrees with independent native measurements")
        if (result != self.environment.result
                or read_json(worker.output / "plan.json") != trace[-1]["action"]["plan"]
                or read_json(worker.output / "public-rollout.json") != trace[-1]["native_rollout"]):
            raise ValueError("Native evidence differs from the dispatched action or observed rollout")
        return dict(outcome="success" if verdict["success"] else "failure",
                    reason="Independent object pose, release and disturbance checks in Isaac Sim",
                    measurements=dict(checks=verdict["checks"], physics_steps=result["physics_steps"]))


class Suite:
    def __init__(self, python, root, *, resource="isaac-gpu-0", gui=False, cache_root=None):
        self.python, self.root, self.resource, self.gui = python, Path(root), resource, gui
        self.cache_root = cache_root
        self._legacy = local()

    def identity(self):
        return dict(kind="isaac-trial-suite/v1", worker=NativeWorker(self.python, self.root, gui=self.gui, cache_root=self.cache_root).identity(),
                    resource=self.resource, scope=SCOPE, execution="native-isaacsim-5.1")

    def admit(self, candidate):
        from .harness import Evaluation
        return Evaluation(self.python, self.root, self.gui).admit(candidate, None)

    def environment(self, task):
        if task != TASK:
            raise ValueError("Unknown Isaac task")
        self._legacy.current = Environment(NativeWorker(self.python, self.root, gui=self.gui, cache_root=self.cache_root), self.resource)
        return self._legacy.current

    def trial(self, candidate, task):
        if task != TASK:
            raise ValueError("Unknown Isaac task")
        environment = Environment(NativeWorker(self.python, self.root, gui=self.gui, cache_root=self.cache_root), self.resource)
        return TrialAdapters(environment, Policy(candidate, environment), Verifier(environment))

    def policy(self, candidate, task):
        return Policy(candidate, self._legacy.current)

    def verifier(self, task):
        return Verifier(self._legacy.current)
