"""Development-only public proprioception and action trace.

The policy engine binds this recorder in its source closure. Callers must supply
trusted development attribution before using it. A trace is diagnostic evidence,
never a native score receipt.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .observations import PROPRIO, validate_action
from PhysicalRSI_core.infra.storage import digest, file_digest, read_json

TRACE_SCOPE = "Public proprioception and issued commands; not ground-truth task state or score"


def validate_binding(binding, *, candidate_sha256, task):
    """Check a caller-supplied development case against its frozen protocol."""
    required = {"schema", "split", "candidate_sha256", "task", "case", "protocol_path", "protocol_sha256"}
    if (not isinstance(binding, dict) or set(binding) != required
            or binding["schema"] != "physicalrsi.behavior-development-binding/v1"
            or binding["split"] != "development"
            or binding["candidate_sha256"] != candidate_sha256 or binding["task"] != task):
        raise ValueError("Invalid development diagnostic binding")
    protocol = read_json(binding["protocol_path"])
    if (digest(protocol) != binding["protocol_sha256"]
            or protocol.get("schema") != "physicalrsi.case-protocol/v1"
            or protocol.get("identity_keys") != ["instance_id"]):
        raise ValueError("Development protocol changed or uses a different reset identity")
    case = binding["case"]
    if (not isinstance(case, dict) or case not in protocol["splits"]["development"].get(task, [])
            or case.get("task") != task or case.get("mode") != "train" or case.get("rollout_id") != 0
            or type(case.get("instance_id")) is not int or case["instance_id"] < 0
            or case.get("instance_index") != case["instance_id"]):
        raise ValueError("Diagnostic case is not a declared native development reset")
    return digest([task, {"instance_id": case["instance_id"]}])


class DevelopmentTrace:
    def __init__(self, path, *, split, candidate_sha256, task, case_id):
        if split != "development":
            raise ValueError("Only development episodes may produce proposal diagnostics")
        if (len(candidate_sha256) != 64
                or any(c not in "0123456789abcdef" for c in candidate_sha256)):
            raise ValueError("Declare a candidate SHA256")
        if not isinstance(task, str) or not task or not isinstance(case_id, str) or not case_id:
            raise ValueError("Declare the task and development case identity")
        self.path = Path(path)
        self._file = self.path.open("xb")
        self._digest = hashlib.sha256()
        self._steps = 0
        self._batch = None
        self._task = task
        self._candidate_sha256 = candidate_sha256
        self._case_id = case_id
        self._closed = False
        self._write(dict(schema="physicalrsi.behavior-development-trace/v1",
                         split=split, candidate_sha256=candidate_sha256,
                         task=task, case_id=case_id,
                         scope=TRACE_SCOPE))

    def identity(self):
        return dict(schema="physicalrsi.behavior-development-trace/v1", split="development",
                    candidate_sha256=self._candidate_sha256, task=self._task,
                    case_id=self._case_id, path=str(self.path.resolve()))

    def _write(self, row):
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        self._file.write(data)
        self._file.flush()
        self._digest.update(data)

    def append(self, public, *, primitive_actions, issued_action):
        """Record commands without changing them; no controller-state introspection."""
        if self._closed:
            raise ValueError("Trace is closed")
        if public.get("task") != self._task or set(public["proprio"]) != set(PROPRIO):
            raise ValueError("Expected the attributed public proprioception contract")
        values = {}
        batch = None
        for name, indices in PROPRIO.items():
            value = np.asarray(public["proprio"][name])
            if (value.ndim != 2 or value.shape[1] != len(indices) or value.shape[0] < 1
                    or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()):
                raise ValueError("Malformed public proprioception")
            if batch is not None and value.shape[0] != batch:
                raise ValueError("Inconsistent observation batch")
            batch = value.shape[0]
            values[name] = value.tolist()
        if self._batch is not None and self._batch != batch:
            raise ValueError("Episode batch size changed")
        issued = validate_action(issued_action, batch_size=batch).tolist()
        primitives = [validate_action(action, batch_size=batch).tolist() for action in primitive_actions]
        self._write(dict(step=self._steps, proprio=values,
                         primitive_actions=primitives, issued_action=issued))
        self._batch = batch
        self._steps += 1

    def close(self):
        """Return a content digest; this does not certify episode completion."""
        if not self._closed:
            self._file.close()
            self._closed = True
        return dict(path=str(self.path), sha256=self._digest.hexdigest(),
                    recorded_steps=self._steps, qualification=None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def verify_trace(path, *, expected_identity, expected_steps):
    """Validate a complete public-only trace before admitting it as feedback."""
    if type(expected_steps) is not int or expected_steps < 1:
        raise ValueError("Declare the native completed step count")
    header_expected = {k: expected_identity[k] for k in
                       ("schema", "split", "candidate_sha256", "task", "case_id")}
    if header_expected["split"] != "development":
        raise ValueError("Only development diagnostics may enter feedback")
    header_expected["scope"] = TRACE_SCOPE
    sha = file_digest(Path(path))
    count = 0
    with Path(path).open() as stream:
        if json.loads(next(stream)) != header_expected:
            raise ValueError("Diagnostic header differs from policy attribution")
        for line in stream:
            row = json.loads(line)
            if (set(row) != {"step", "proprio", "primitive_actions", "issued_action"}
                    or type(row["step"]) is not int or row["step"] != count
                    or set(row["proprio"]) != set(PROPRIO)
                    or not isinstance(row["primitive_actions"], list)):
                raise ValueError("Unexpected or discontinuous diagnostic fields")
            for name, indices in PROPRIO.items():
                values = np.asarray(row["proprio"][name])
                if (values.shape != (1, len(indices)) or not np.issubdtype(values.dtype, np.number)
                        or not np.isfinite(values).all()):
                    raise ValueError("Malformed diagnostic proprioception")
            validate_action(np.asarray(row["issued_action"]), batch_size=1)
            for action in row["primitive_actions"]:
                validate_action(np.asarray(action), batch_size=1)
            count += 1
    if count != expected_steps:
        raise ValueError("Diagnostic count differs from completed native steps")
    if file_digest(Path(path)) != sha:
        raise ValueError("Diagnostic trace changed during verification")
    return dict(sha256=sha, recorded_steps=count, qualification=None)


def seal_trace(source, destination, *, expected_identity, expected_steps):
    """Copy stable public diagnostics into the native receipt evidence folder."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    original_sha = file_digest(source)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        result = verify_trace(temporary, expected_identity=expected_identity, expected_steps=expected_steps)
        if result["sha256"] != original_sha or file_digest(source) != original_sha:
            raise ValueError("Diagnostic trace changed while sealing")
        # The native trial folder is exclusively locked by its owner.
        temporary.rename(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)
