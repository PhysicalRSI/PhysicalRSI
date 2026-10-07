"""PhysicalRSI execution lifecycle derived from upstream Toolkit dispatch.

Record before invocation, preserve action and observation errors independently,
and retain exclusive effects until the operation actually returns.
"""

import time
import uuid
import hashlib
import os
import stat
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Lock

from PhysicalRSI_core.contracts import Cancelled

from .artifacts import Artifacts
from .events import Event, NullEvents
from .storage import atomic_json, identifier, strict_json


class Execution:
    def __init__(self, root, *, observe=None, events=None):
        self.root = Path(root).resolve()
        self.observe = observe
        self.events = events or NullEvents()
        self._condition = Condition()
        self._effects = {}
        self._active = {}
        self.artifacts = Artifacts(self.root / "artifacts")

    def read(self, episode, call_id, *, max_record_bytes=1024 * 1024,
             max_artifact_bytes=64 * 1024 * 1024, expected_sha256=None):
        """Inspect a bounded journal snapshot and its unique local artifacts.

        This does not retry an operation or reinterpret its outcome. Supply an
        independently retained record digest when checking an earlier snapshot.
        """
        if any(type(limit) is not int or limit < 0 for limit in (max_record_bytes, max_artifact_bytes)):
            raise ValueError("Execution inspection requires nonnegative byte allowances")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("Expected execution digest must be a SHA256")
        folder = self.root / identifier(episode)
        path = folder / (identifier(call_id) + ".json")
        if folder.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Execution journal must be a physical regular file")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_record_bytes:
                raise ValueError("Execution journal exceeds its byte allowance or is not regular")
            payload = stream.read(max_record_bytes + 1)
        if len(payload) > max_record_bytes:
            raise ValueError("Execution journal exceeds its byte allowance")
        sha = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError("Execution journal differs from the expected snapshot")
        record = strict_json(payload)
        if (not isinstance(record, dict) or record.get("id") != call_id or record.get("episode") != episode
                or record.get("state") not in {"started", "completed", "failed", "cancelled", "uncertain"}
                or "input" not in record or (record["state"] == "completed" and "output" not in record)):
            raise ValueError("Execution record identity, state or payload is invalid")
        references, remaining = {}, max_artifact_bytes

        def verify(value):
            nonlocal remaining
            if isinstance(value, dict):
                if "artifact" in value and {"sha256", "bytes"}.intersection(value):
                    name = value["artifact"]
                    if not isinstance(name, str):
                        raise ValueError("Artifact location must be a string")
                    if name in references:
                        if value != references[name]:
                            raise ValueError("Conflicting references to one execution artifact")
                    else:
                        content = self.artifacts.read(value, max_bytes=remaining)
                        remaining -= len(content)
                        references[name] = dict(value)
                else:
                    for item in value.values():
                        verify(item)
            elif isinstance(value, list):
                for item in value:
                    verify(item)

        for field in ("input", "output", "observation"):
            if field in record:
                verify(record[field])
        return dict(schema="physicalrsi.execution-inspection/v1", record=record, record_sha256=sha,
                    artifacts=[references[name] for name in sorted(references)],
                    artifact_bytes=max_artifact_bytes - remaining)

    def cancel_and_wait(self, episode, timeout=5):
        until = time.monotonic() + timeout
        with self._condition:
            for context in self._active.values():
                if context.episode == episode:
                    context.cancelled.set()
            while any(context.episode == episode for context in self._active.values()):
                left = until - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Operation still active; effects remain owned")
                self._condition.wait(left)

    def __call__(self, operation, value, context):
        context.check()
        call_id = uuid.uuid4().hex
        path = self.root / context.episode / (call_id + ".json")
        record = dict(
            id=call_id,
            episode=context.episode,
            operation=operation.name,
            revision=operation.revision,
            harness_revision=context.harness_revision,
            input_contract=asdict(operation.input),
            output_contract=asdict(operation.output),
            effects=sorted(operation.effects),
            state="started",
        )
        started = time.monotonic()
        with self._condition:
            self._active[call_id] = context
            locks = [
                self._effects.setdefault(effect, Lock())
                for effect in sorted(operation.effects)
            ]
        error = None
        result = None
        try:
            record["input"] = self.artifacts.encode(value)
            atomic_json(path, record)
            self.events.emit(Event("operation.started", dict(record)))
            with ExitStack() as stack:
                # Composite nodes carry union effects for planning. Their leaves
                # acquire the actual locks, so nested parallel work cannot deadlock.
                if not operation.children:
                    for lock in locks:
                        while not lock.acquire(timeout=0.05):
                            context.check()
                        stack.callback(lock.release)
                context.check()
                try:
                    result = operation.call(value, context)
                    record["output"] = self.artifacts.encode(result)
                    context.check()
                    record["state"] = "completed"
                except BaseException as caught:
                    error = caught
                    record["state"] = (
                        "cancelled" if isinstance(caught, Cancelled) else "failed"
                    )
                    record["error"] = type(caught).__name__ + ": " + str(caught)
                    if request_id := getattr(caught, "request_id", None):
                        record["request_id"] = request_id
                        record["state"] = "uncertain"
                # Observe even after failure/cancellation, while effects are held.
                if operation.effects and not operation.children and self.observe:
                    try:
                        record["observation"] = self.artifacts.encode(
                            self.observe(operation, context)
                        )
                    except Exception as caught:
                        record["observation_error"] = str(caught)
                        if error is None:
                            error = caught
                            record["state"] = "uncertain"
                record["elapsed_seconds"] = time.monotonic() - started
                atomic_json(path, record)
                self.events.emit(Event("operation.finished", dict(record)))
            if error is not None:
                raise error
            return result
        finally:
            with self._condition:
                self._active.pop(call_id, None)
                self._condition.notify_all()
