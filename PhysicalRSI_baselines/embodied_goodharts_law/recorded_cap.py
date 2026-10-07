"""Record CAP requests before native effects in ExperimentRuntime.

The isolated program's callbacks enqueue data only. The experiment thread owns
all simulator calls through RecordedPrimitiveEnvironment.step. Official proxy
measurement stays on that thread and is never returned to the candidate.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.embodiment import Embodiment, System1
from PhysicalRSI_core.infra.storage import canonical, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_baselines.robodojo.code_execution import execute_policy
from .candidate_program import bind_candidate


def _copy(value):
    import json
    return json.loads(canonical(value))


class RecordedCapPolicy:
    """Convert isolated CAP callbacks into experiment actions, without actuation."""

    def __init__(self, candidate, *, specification, capabilities, runtime, output,
                 max_calls=100, teardown_seconds=5):
        self.candidate = deepcopy(candidate)
        self.bound = bind_candidate(candidate)
        if not isinstance(specification, System1) or specification.revision != self.bound.harness_sha256:
            raise ValueError("System 1 must match the frozen candidate")
        if (type(max_calls) is not int or max_calls < 1 or
                type(teardown_seconds) not in (int, float) or not 0 < teardown_seconds <= 30):
            raise ValueError("Declare bounded call and teardown limits")
        self.capabilities = tuple(sorted(capabilities))
        if (not self.capabilities or len(set(self.capabilities)) != len(self.capabilities) or
                any(not isinstance(name, str) or not name for name in self.capabilities)):
            raise ValueError("Declare explicit native capabilities")
        self.specification, self.runtime, self.output = specification, Path(runtime), Path(output)
        self.max_calls, self.teardown_seconds = max_calls, teardown_seconds
        self._runtime_sha = digest(read_json(self.runtime / "manifest.json"))
        self._worker = None

    def describe(self):
        return self.specification

    def identity(self):
        if verify_harness(self.candidate) != self.bound.harness_sha256:
            raise ValueError("Executing candidate changed")
        if digest(read_json(self.runtime / "manifest.json")) != self._runtime_sha:
            raise ValueError("Isolation runtime manifest changed")
        return dict(kind="egl-recorded-cap-policy/v1", binding=self.bound.identity(),
                    specification=asdict(self.specification), capabilities=self.capabilities,
                    runtime_sha256=self._runtime_sha, max_calls=self.max_calls,
                    teardown_seconds=self.teardown_seconds, implementation=file_digest(Path(__file__)))

    def begin_episode(self, task, case, observation, context):
        context.check()
        if self._worker is not None or context.deadline is None:
            raise ValueError("Use a fresh policy with a finite episode deadline")
        if context.harness_revision != self.bound.harness_sha256:
            raise ValueError("Episode belongs to a different candidate")
        self._cancel = Event()
        self._events, self._replies = Queue(maxsize=1), Queue(maxsize=1)
        self._pending = None
        self._request_id = 0
        self._finished = False
        self._episode = context.episode

        def callback(name):
            def call(args, kwargs, *, deadline):
                index = self._request_id
                self._request_id += 1
                self._events.put_nowait(_copy(dict(kind="call", id=index, method=name, args=args, kwargs=kwargs)))
                while not self._cancel.is_set():
                    if monotonic() >= deadline:
                        raise TimeoutError("CAP response deadline reached")
                    try:
                        response = self._replies.get(timeout=min(.02, max(.001, deadline - monotonic())))
                        if response["id"] != index:
                            raise ValueError("CAP response identity mismatch")
                        return response["value"]
                    except Empty:
                        pass
                raise RuntimeError("CAP execution cancelled")
            return call

        def run():
            try:
                value = execute_policy(self.bound.source, self.bound.memory,
                    handlers={name: callback(name) for name in self.capabilities},
                    runtime=self.runtime, output=self.output / context.episode,
                    deadline=context.deadline, timeout_s=max(.001, context.deadline - monotonic()),
                    cancelled=self._cancel, max_calls=self.max_calls,
                    expected_runtime_sha256=self._runtime_sha, output_bytes=32 * 1024 * 1024)
                self._events.put_nowait(_copy(dict(kind="finish", value=value)))
            except BaseException as error:
                if not self._cancel.is_set():
                    self._events.put_nowait(dict(kind="error", error_type=type(error).__name__))

        self._worker = Thread(target=run, daemon=True)
        self._worker.start()

    def act(self, observation, context):
        context.check()
        if context.episode != self._episode or self._finished:
            raise ValueError("CAP policy is not active for this episode")
        if self._pending is not None:
            reply = observation.get("reply")
            if not isinstance(reply, dict) or set(reply) != {"id", "value"} or reply["id"] != self._pending:
                raise ValueError("Missing matching primitive response")
            # Deliberately forward only this public response, never evaluation.
            self._replies.put_nowait(_copy(reply))
            self._pending = None
        while True:
            context.check()
            try:
                event = self._events.get(timeout=.02)
            except Empty:
                if not self._worker.is_alive():
                    raise RuntimeError("Isolated CAP worker exited without a result")
                continue
            if event["kind"] == "error":
                raise RuntimeError("Isolated CAP failed: " + event["error_type"])
            if event["kind"] == "finish":
                self._finished = True
            else:
                self._pending = event["id"]
            return event

    def end_episode(self, context):
        if self._worker is None:
            return {"stopped": True, "started": False}
        self._cancel.set()
        self._worker.join(self.teardown_seconds)
        if self._worker.is_alive():
            raise ReconciliationRequired("CAP worker did not confirm shutdown")
        return {"stopped": True, "program_returned": self._finished}


class RecordedPrimitiveEnvironment:
    """Own native effects, public replies, and separately recorded proxy evidence.

    factory(case) returns (primitive_api, measurement_callback, close_callback).
    Construction/identity must have no simulator effects. The caller must close
    the environment in finally, including after an interrupted experiment. An
    outer process deadline remains required for blocking native calls.
    """

    def __init__(self, *, specification, identity, factory):
        if not isinstance(specification, Embodiment) or specification.mode != "simulation":
            raise ValueError("Declare the simulation embodiment")
        self.specification, self._identity = specification, _copy(identity)
        self.factory = factory
        self._close = None
        self._episode = None

    def describe(self):
        return self.specification

    def identity(self):
        return dict(kind="egl-recorded-native-environment/v1", native=self._identity,
                    specification=asdict(self.specification), implementation=file_digest(Path(__file__)))

    def _observation(self, reply):
        measurements = _copy(self._measure())
        if type(measurements.get("official_success")) is not bool:
            raise ValueError("Native official success must be a Boolean")
        return {"reply": reply, "evaluation": measurements}

    def reset(self, case, context):
        context.check()
        if self._episode is not None:
            raise ValueError("Create a fresh environment for each experiment")
        self._episode = context.episode
        self._api, self._measure, self._close = self.factory(_copy(case))
        self._handlers = self._api.handlers
        self._next = 0
        self._finished = False
        observation = self._observation(None)
        context.check()
        # An already-satisfied reset does not establish task execution success.
        return {"ready": not observation["evaluation"]["official_success"], "observation": observation}

    def step(self, action, context):
        context.check()
        if context.episode != self._episode or self._finished:
            raise ValueError("Native episode is not active")
        if action.get("kind") == "finish" and set(action) == {"kind", "value"}:
            self._finished = True
            observation = self._observation(None)
            return {"observation": observation, "terminated": True}
        if (set(action) != {"kind", "id", "method", "args", "kwargs"} or action["kind"] != "call"
                or type(action["id"]) is not int or action["id"] != self._next
                or action["method"] not in self._handlers or not isinstance(action["args"], list)
                or not isinstance(action["kwargs"], dict)):
            raise ValueError("Invalid recorded primitive request")
        if context.deadline is None:
            raise ValueError("Native calls require an episode deadline")
        self._next += 1
        value = self._handlers[action["method"]](action["args"], action["kwargs"], deadline=context.deadline)
        observation = self._observation(dict(id=action["id"], value=_copy(value)))
        context.check()
        self._finished = observation["evaluation"]["official_success"]
        return {"observation": observation, "terminated": self._finished}

    def quiesce(self, context):
        context.check()
        self.close()
        context.check()
        return {"quiescent": True}

    def close(self):
        if self._close is not None:
            close, self._close = self._close, None
            close()


class OfficialProxyVerifier:
    """Report the native predicate only; this is not an independent task audit."""
    def identity(self):
        return dict(kind="egl-native-official-proxy/v1", implementation=file_digest(Path(__file__)))

    def verify(self, case, trace):
        if not trace:
            raise ValueError("Official verification requires a recorded trajectory")
        result = trace[-1]["observation"]["evaluation"]
        success = result["official_success"]
        if type(success) is not bool:
            raise ValueError("Official predicate missing or invalid")
        return {"outcome": "success" if success else "failure",
                "reason": "Native official predicate; independent intended-task audit not performed",
                "measurements": {**_copy(result), "independent_audit_performed": False}}
