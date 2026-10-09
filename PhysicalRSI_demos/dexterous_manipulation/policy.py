"""Bridge the unchanged Galbot agent event loop into Core's step lifecycle."""
import copy
import queue
import subprocess
import threading
from pathlib import Path

from PhysicalRSI_core.embodiment import System1, Dependency
from hybrid_rollout.robodojo.io import InputError
from hybrid_rollout.robodojo.skill.run import CodexPolicy
from hybrid_rollout.robodojo.skill.transport import StdioAppServer

from .common import Rejected, digest, file_hash, implementation_hash


class AgentTools:
    def validate(self, action, observation, limits):
        raise NotImplementedError("Environment adapter must validate its action contract")

    def __init__(self, bridge, task, initial):
        self.bridge, self.task = bridge, task
        self.phase, self.tick, self.counters = "start", 0, dict(actions=0)
        self.request = copy.deepcopy(initial)
        self.observation_path = initial["observation_path"]

    def handlers(self):
        raise NotImplementedError("Environment adapter must supply its action tools")

    def next_call(self):
        raise NotImplementedError("Environment adapter must supply its next tool call")

    def start(self, task, output_dir):
        if self.phase != "start" or task != self.task:
            raise InputError("Read the prepared task exactly once; no second reset is available")
        self.phase = "act"
        return copy.deepcopy(self.request)

    def _action(self, response):
        if self.phase != "act":
            raise InputError("Start the prepared episode first, and do not act after finishing")
        try:
            action = self.validate(response, self.request, self.request["limits"])
        except Rejected as error:
            raise InputError(str(error)) from error
        self.bridge.actions.put(action)
        while True:
            if self.bridge.stopped.is_set():
                raise RuntimeError("Core ended the policy episode")
            try:
                packet = self.bridge.feedback.get(timeout=.05)
                break
            except queue.Empty:
                pass
        if isinstance(packet, BaseException):
            raise packet
        self.request = packet
        self.observation_path = packet["observation_path"]
        self.tick += 1
        self.counters["actions"] += 1
        if packet.get("rollout_finished"):
            self.phase = "done"
        return copy.deepcopy(packet)

    def act(self, observation_path, response, output_dir):
        # Host owns immutable evidence paths; agent-provided output_dir is only
        # accepted for upstream call compatibility, never used as a write path.
        if str(observation_path) != self.observation_path:
            raise InputError("Use the latest host observation_path")
        return self._action(response)


class GalbotPolicy:
    def __init__(self, environment, profile, workspace, *, codex="codex", timeout=120., worker_factory=None):
        self.environment, self.profile = environment, profile
        self.workspace, self.codex, self.timeout = Path(workspace), codex, timeout
        self.worker_factory = worker_factory
        self.actions, self.feedback = queue.Queue(), queue.Queue()
        self.stopped = threading.Event()
        self.worker_done = threading.Event()
        self.thread = None
        self.controller = self.transport = None
        self.error = None
        self.awaiting_feedback = False

    def identity(self):
        return dict(name=getattr(self.profile, "policy_name", "galbot-astra-dexterous"), implementation=implementation_hash(), profile=self.profile.identity())

    def describe(self):
        interface = self.environment.describe()
        dependencies=[Dependency("gpt-6-astra", "gpt-6-astra", "model"),
                      Dependency(self.profile.skill_root.name,self.profile.identity()["skill_sha256"],"skill")]
        if self.profile.memory:
            dependencies.append(Dependency("candidate-memory",file_hash(self.profile.memory),"memory"))
        candidate=getattr(self.profile,"candidate",None)
        if candidate:
            dependencies.extend(Dependency(name,sha,"tool") for name,sha in candidate["components"]["control"].items()
                                if name.startswith("programs/"))
        return System1(name=self.identity()["name"], revision=digest(self.identity()),
            observation=interface.observation, action=interface.action,
            dependencies=tuple(dependencies),timing=interface.timing)

    def _transport(self, argv, workspace):
        env = self.profile.child_environment(workspace)
        transport = StdioAppServer(argv, workspace,
            popen=lambda *a, **kw:subprocess.Popen(*a, **kw, env=env))
        self.transport = transport
        if self.stopped.is_set():
            transport.close()
            raise RuntimeError("Episode ended during agent startup")
        return transport

    def begin_episode(self, task, case, observation, context):
        self.tools = getattr(self.profile, "tools_class", AgentTools)(self, task, observation)
        def run():
            try:
                if self.worker_factory:
                    self.worker_factory(self.tools)
                else:
                    self.controller = CodexPolicy(self.workspace, self.codex, timeout=self.timeout,
                        method="gpt_only", runtime=self.profile, transport_factory=self._transport)
                    if not self.stopped.is_set():
                        self.controller.run(self.tools)
            except BaseException as error:
                self.error = error
            finally:
                self.worker_done.set()
        self.thread = threading.Thread(target=run, name="galbot-policy", daemon=True)
        self.thread.start()

    def act(self, observation, context):
        if self.awaiting_feedback:
            self.feedback.put(copy.deepcopy(observation))
            self.awaiting_feedback = False
        while True:
            context.check()
            if self.error:
                raise RuntimeError("Galbot policy failed") from self.error
            try:
                action = self.actions.get(timeout=.05)
                self.awaiting_feedback = True
                return action
            except queue.Empty:
                if self.worker_done.is_set():
                    if self.error:
                        raise RuntimeError("Galbot policy failed") from self.error
                    raise RuntimeError("Agent ended before the environment terminated")

    def end_episode(self, context):
        if self.thread is None:
            return dict(stopped=True, started=False)
        # Quiesce the environment even when the Core context deadline has expired.
        quiet = self.environment.stop_only()
        if self.awaiting_feedback and not self.stopped.is_set():
            packet = dict(self.environment.latest, rollout_finished=True)
            self.feedback.put(packet)
            self.awaiting_feedback = False
        # Give the tool reply time to be persisted; no extra model turn needed.
        self.worker_done.wait(.3)
        self.stopped.set()
        if self.transport is not None:
            self.transport.close()
        self.thread.join(timeout=8.)
        if self.thread.is_alive():
            raise RuntimeError("Agent worker termination is unconfirmed")
        self.profile.cleanup_workspace(self.workspace)
        return dict(stopped=True, device_stop=quiet)
