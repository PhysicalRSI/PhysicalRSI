"""Frozen System 1 program; System 2 materializes a new candidate between trials.

Programs are reviewed Python, not sandboxed untrusted code. They receive only
projected onboard observations and a fixed action-backend port. Simulator and
scoring objects are never provided through this interface.
"""
from copy import deepcopy
from pathlib import Path
import types

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.embodiment import System1
from PhysicalRSI_core.infra.storage import file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from .observations import project, validate_action
from .diagnostics import DevelopmentTrace
from . import engine

OBSERVATION = Contract("behavior-r1pro-onboard-v1", embodiment="R1Pro")
ACTION = Contract("behavior-r1pro-native-controls-v1", embodiment="R1Pro")


class ProgramPolicy:
    """Load one verified control.py and execute its predefined feedback logic.

    A program defines initialize(task) -> state and step(observation, state,
    primitive) -> action. primitive(observation) calls the fixed action backend.
    The backend owns model inference; there is no code-generation port here.
    """
    def __init__(self, candidate, backend, *, depth=False, diagnostics=None):
        self.candidate = deepcopy(candidate)
        self.revision = verify_harness(candidate)
        self.engine = engine.verify(self.candidate)
        self.backend = backend
        self.backend_identity = deepcopy(backend.identity())
        backend_path = Path(candidate["root"]) / "backend.json"
        if (candidate["components"]["foundation"].get("backend.json") != file_digest(backend_path)
                or read_json(backend_path) != self.backend_identity):
            raise ValueError("Bind the fixed action backend in the foundation closure")
        self.depth = depth
        if diagnostics is not None:
            if not isinstance(diagnostics, DevelopmentTrace):
                raise ValueError("Expected a development-only diagnostic recorder")
            if diagnostics.identity()["candidate_sha256"] != self.revision:
                raise ValueError("Diagnostic attribution differs from candidate")
        self.diagnostics = diagnostics
        self._diagnostic_episode_started = False
        self.path = Path(candidate["root"]) / "control.py"
        if candidate["components"]["control"].get("control.py") != file_digest(self.path):
            raise ValueError("Declare the executable control.py in the candidate closure")
        self._program = None
        self._state = None
        self._task = None

    def identity(self):
        self._stable()
        return dict(kind="behavior-fixed-program/v1", revision=self.revision,
                    engine=self.engine,
                    backend=self.backend_identity, depth=self.depth,
                    diagnostics=self.diagnostics.identity() if self.diagnostics is not None else None,
                    implementation=file_digest(Path(__file__)),
                    observation_projection=file_digest(Path(__file__).with_name("observations.py")))

    def _stable(self):
        if engine.verify(self.candidate) != self.engine:
            raise ValueError("Shared System 1 implementation changed during an episode")
        if verify_harness(self.candidate) != self.revision or self.backend.identity() != self.backend_identity:
            raise ValueError("Program or action backend changed during an episode")

    def describe(self):
        return System1("behavior-fixed-program", self.revision, OBSERVATION, ACTION)

    def begin_episode(self, task, case, observation, context):
        context.check()
        self._stable()
        if context.harness_revision and context.harness_revision != self.revision:
            raise ValueError("Executing candidate differs from the core revision")
        if self.diagnostics is not None:
            if self.diagnostics.identity()["task"] != task or self._diagnostic_episode_started:
                raise ValueError("A diagnostic recorder binds exactly one development episode")
            self._diagnostic_episode_started = True
        # A new module namespace prevents state from leaking between episodes.
        module = types.ModuleType("behavior_candidate_" + self.revision)
        exec(compile(self.path.read_bytes(), str(self.path), "exec"), module.__dict__)
        self._stable()
        if not callable(getattr(module, "initialize", None)) or not callable(getattr(module, "step", None)):
            raise ValueError("Program must define initialize and step")
        self._task = task
        self._program = module
        self._state = module.initialize(task)
        self.backend.reset(task)
        context.check()

    def act(self, observation, context):
        context.check()
        self._stable()
        if self._program is None:
            raise ValueError("Initialize an episode before acting")
        public = project(observation, task=self._task, depth=self.depth)
        batch_size = public["proprio"]["base_qvel"].shape[0]
        diagnostic_public = (dict(task=self._task, proprio=deepcopy(public["proprio"]))
                             if self.diagnostics is not None else None)
        primitive_actions = []

        def primitive(onboard):
            context.check()
            result = self.backend.act(deepcopy(onboard), context)
            context.check()
            action = validate_action(result, batch_size=batch_size)
            if self.diagnostics is not None:
                primitive_actions.append(action.copy())
            return action

        result = self._program.step(public, self._state, primitive)
        context.check()
        self._stable()
        # ExperimentRuntime persists JSON actions. The websocket transport can
        # reconstruct an ndarray without altering values or controller units.
        action = validate_action(result, batch_size=batch_size)
        if self.diagnostics is not None:
            self.diagnostics.append(diagnostic_public, primitive_actions=primitive_actions,
                                    issued_action=action)
        return action.tolist()

    def end_episode(self, context):
        self._program = self._state = self._task = None
        # The policy owns no child worker; the service supervisor owns backend
        # lifecycle. reset() at the next episode clears backend episode history.
        result = {"stopped": True, "owned_workers": 0}
        if self.diagnostics is not None:
            result["diagnostics"] = self.diagnostics.close()
        return result
