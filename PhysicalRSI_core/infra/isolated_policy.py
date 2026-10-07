"""Persistent isolated System 1 code which returns actions through a data port.

There are no robot callbacks, service credentials, network sockets or host
paths in the child. The trusted experiment owns dispatch, verification and stop.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

from ..contracts import ReconciliationRequired
from ..embodiment import System1
from ..timing import finite_seconds
from .isolated_program import PythonIsolation
from .storage import atomic_json, canonical, digest, file_digest, identifier, read_json


_RUNNER = r'''
import json
import time

def publish(value):
    with open('/output/result.json', 'w') as stream:
        json.dump(value, stream, allow_nan=False)

index = 0
def exchange(value):
    global index
    current = index
    index += 1
    publish(dict(kind='call', id=current, method='exchange', args=[value], kwargs={}))
    while True:
        with open('/response.json') as stream:
            reply = json.load(stream)
        if reply.get('id') == current:
            return reply['value']
        time.sleep(.005)

inputs = json.load(open('/input.json'))
scope = {}
exec(compile(inputs['source'], '<isolated-system1>', 'exec'), scope)
request = exchange(dict(ready=True))
while request['operation'] != 'close':
    if request['operation'] == 'begin':
        data = request['payload']
        state = scope['begin_episode'](data['task'], data['case'], data['observation'])
        value = None
    elif request['operation'] == 'act':
        result = scope['act'](request['payload'], state)
        if not isinstance(result, dict) or set(result) != {'action', 'state'}:
            raise ValueError('act must return action and transient episode state')
        value, state = result['action'], result['state']
    else:
        raise ValueError('Unknown data operation')
    request = exchange(dict(id=request['id'], value=value))
publish(dict(kind='result', value=dict(closed=True)))
'''


class IsolatedPolicy:
    def __init__(self, source, *, specification, isolation, output, max_actions=100, call_seconds=2):
        if not isinstance(source, str) or not source.strip() or not isinstance(specification, System1):
            raise ValueError("Declare System 1 source and exact observation/action contracts")
        if not isinstance(isolation, PythonIsolation) or type(max_actions) is not int or max_actions <= 0:
            raise ValueError("A PythonIsolation profile and positive action budget are required")
        finite_seconds(call_seconds)
        if len(source.encode()) > isolation.message_bytes:
            raise ValueError("Isolated policy source exceeds the message allowance")
        self.source, self.specification, self.isolation = source, specification, isolation
        self.output = Path(output).resolve()
        self.max_actions, self.call_seconds = max_actions, call_seconds
        self._frozen = deepcopy(self.identity())
        self._worker, self._session, self._cleanup = None, None, None
        self._lock = Lock()

    def identity(self):
        return dict(kind="isolated-system1/v1", source_sha256=digest(self.source),
                    specification=asdict(self.specification), isolation=self.isolation.identity(),
                    max_actions=self.max_actions, call_seconds=self.call_seconds,
                    implementation=file_digest(Path(__file__)))

    def describe(self):
        return self.specification

    def _stable(self):
        if self.identity() != self._frozen:
            raise ValueError("Frozen isolated policy changed")

    def _exchange(self, args, kwargs, *, deadline):
        if kwargs or not isinstance(args, list) or len(args) != 1 or not isinstance(args[0], dict):
            raise ValueError("Invalid isolated policy exchange")
        value = args[0]
        if self._pending is None:
            if self._ready or value != dict(ready=True):
                raise ValueError("Unexpected isolated policy greeting")
            self._ready = True
        else:
            if set(value) != {"id", "value"} or type(value["id"]) is not int or value["id"] != self._pending:
                raise ValueError("Isolated policy response does not match the pending request")
            self._replies.put_nowait(value)
        while not self._cancel.is_set():
            if monotonic() >= deadline:
                raise TimeoutError("Isolated policy episode deadline reached")
            try:
                request = self._requests.get(timeout=min(.02, max(.001, deadline - monotonic())))
                self._pending = request["id"]
                return request
            except Empty:
                pass
        raise RuntimeError("Isolated policy was cancelled")

    def _invoke(self, operation, payload, context):
        context.check()
        self._stable()
        if context.episode != self._session or context.harness_revision != self.specification.revision:
            raise ValueError("Isolated policy call belongs to a different frozen episode")
        if context.deadline is None:
            raise ValueError("Isolated policy calls require a bounded execution context")
        finite_seconds(context.deadline - monotonic())
        request = dict(id=self._index, operation=operation, payload=payload)
        encoded = canonical(request)
        if len(encoded) > self.isolation.message_bytes:
            raise ValueError("Policy input exceeds the message allowance")
        self._requests.put_nowait(json.loads(encoded))
        deadline = min(context.deadline, self._deadline, monotonic() + self.call_seconds)
        while True:
            context.check()
            if monotonic() >= deadline:
                raise TimeoutError("Isolated policy call deadline reached")
            try:
                response = self._replies.get(timeout=min(.02, max(.001, deadline - monotonic())))
                if response["id"] != self._index:
                    raise ValueError("Unexpected isolated policy response sequence")
                self._index += 1
                context.check()
                if monotonic() >= deadline:
                    raise TimeoutError("Isolated policy response arrived after its call deadline")
                return json.loads(canonical(response["value"]))
            except Empty:
                if not self._worker.is_alive():
                    raise RuntimeError("Isolated policy worker stopped before returning a response")

    def begin_episode(self, task, case, observation, context):
        with self._lock:
            context.check()
            self._stable()
            if context.deadline is None or context.harness_revision != self.specification.revision:
                raise ValueError("Isolated System 1 requires a bounded frozen execution context")
            finite_seconds(context.deadline - monotonic())
            if self._worker is not None and self._cleanup is None:
                raise ReconciliationRequired("Previous isolated policy episode has not been closed")
            directory = self.output / identifier(context.episode)
            if directory.exists():
                raise ReconciliationRequired("An isolated policy episode cannot be replayed")
            self._session, self._directory, self._deadline = context.episode, directory, context.deadline
            self._requests, self._replies = Queue(maxsize=1), Queue(maxsize=1)
            self._cancel, self._cleanup = Event(), None
            self._pending, self._ready, self._index, self._actions = None, False, 0, 0
            self._worker_error = None

            def execute():
                try:
                    self.isolation.run(_RUNNER, dict(source=self.source), output=self._directory,
                        deadline=self._deadline, cancelled=self._cancel,
                        handlers={"exchange": self._exchange}, max_calls=self.max_actions + 2)
                except BaseException as error:
                    self._worker_error = type(error).__name__

            self._worker = Thread(target=execute, name="isolated-system1", daemon=True)
            self._worker.start()
            try:
                self._invoke("begin", dict(task=task, case=case, observation=observation), context)
            except BaseException:
                self._cancel.set()
                self._stop()
                raise

    def act(self, observation, context):
        with self._lock:
            if self._worker is None or self._cleanup is not None or self._actions >= self.max_actions:
                raise ValueError("Isolated policy episode is absent, closed or out of actions")
            try:
                result = self._invoke("act", observation, context)
                self._actions += 1
                return result
            except BaseException:
                self._cancel.set()
                self._stop()
                raise

    def _stop(self):
        allowance = 3 * self.isolation.teardown_seconds + .5
        self._worker.join(allowance)
        if self._worker.is_alive():
            raise ReconciliationRequired("Isolated policy worker teardown is unconfirmed")
        process_path = self._directory / "processes/isolated/process.json"
        process = read_json(process_path) if process_path.exists() else None
        if process is not None and process.get("stopped") is not True:
            raise ReconciliationRequired("Isolated process termination is unconfirmed")
        self._cleanup = dict(stopped=True, episode=self._session, source_sha256=digest(self.source),
                             isolation=self.isolation.identity(), worker=process,
                             worker_error=self._worker_error, actions=self._actions)
        atomic_json(self._directory / "cleanup.json", self._cleanup)
        return deepcopy(self._cleanup)

    def end_episode(self, context):
        with self._lock:
            if self._worker is None:
                return dict(stopped=True, started=False)
            if context.episode != self._session:
                raise ValueError("Cannot close a different isolated policy episode")
            if self._cleanup is not None:
                return deepcopy(self._cleanup)
            if self._worker.is_alive() and not self._cancel.is_set() and monotonic() < self._deadline:
                try:
                    self._requests.put_nowait(dict(id=self._index, operation="close", payload=None))
                except Full:
                    self._cancel.set()
                else:
                    self._worker.join(min(self.isolation.teardown_seconds, max(0, self._deadline - monotonic())))
            self._cancel.set()
            result = self._stop()
            return result
