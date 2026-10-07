"""Bind a declared Python skill artifact to isolated System 1 execution.

Source remains data until the restricted child starts. Admission still owns
the allowed search domain; independent experiment evidence owns selection.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

from ..embodiment import Dependency, System1
from ..infra.isolated_policy import IsolatedPolicy
from ..infra.isolated_program import PythonIsolation
from ..infra.storage import file_digest, relative_path
from .artifacts import verify_harness
from .proposals import strict_json


PYTHON_SKILL_SCHEMA = dict(
    type="object",
    properties=dict(schema=dict(const="physicalrsi.python-skill/v1"),
                    source=dict(type="string", minLength=1, maxLength=100000)),
    required=["schema", "source"], additionalProperties=False)


class IsolatedHarnessPolicy(IsolatedPolicy):
    def __init__(self, candidate, *, skill_file, specification, isolation, output,
                 max_actions=100, call_seconds=2):
        if not isinstance(specification, System1) or not isinstance(isolation, PythonIsolation):
            raise ValueError("Declare System 1 contracts and a PythonIsolation profile")
        self.candidate = deepcopy(candidate)
        revision = verify_harness(self.candidate)
        if specification.revision != revision:
            raise ValueError("System 1 revision differs from the candidate closure")
        if not isinstance(skill_file, str) or relative_path(skill_file).as_posix() != skill_file:
            raise ValueError("Skill file must be a canonical relative path")
        owners = [kind for kind, entries in self.candidate["components"].items() if skill_file in entries]
        if owners != ["skills"]:
            raise ValueError("Executable skill must belong exclusively to the skills component")
        self.skill_file = skill_file
        expected = self.candidate["components"]["skills"][skill_file]
        with (Path(self.candidate["root"]) / skill_file).open("rb") as stream:
            payload = stream.read(isolation.message_bytes + 1)
        if len(payload) > isolation.message_bytes:
            raise ValueError("Python skill artifact exceeds the isolation message allowance")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Python skill changed while being loaded")
        artifact = strict_json(payload)
        if (not isinstance(artifact, dict) or set(artifact) != {"schema", "source"}
                or artifact["schema"] != "physicalrsi.python-skill/v1"
                or not isinstance(artifact["source"], str) or not 1 <= len(artifact["source"]) <= 100000):
            raise ValueError("Invalid Python skill artifact")
        dependency = Dependency(skill_file, expected, "skill")
        existing = [item for item in specification.dependencies if item.kind == "skill" and item.name == skill_file]
        if existing and existing != [dependency]:
            raise ValueError("System 1 skill dependency differs from the artifact")
        specification = replace(specification, dependencies=specification.dependencies + (() if existing else (dependency,)))
        super().__init__(artifact["source"], specification=specification, isolation=isolation,
                         output=output, max_actions=max_actions, call_seconds=call_seconds)

    def identity(self):
        return dict(super().identity(), kind="isolated-harness-policy/v1",
                    harness_sha256=verify_harness(self.candidate), skill_file=self.skill_file,
                    binding_implementation=file_digest(Path(__file__)),
                    serialization=file_digest(Path(__file__).parents[1] / "infra/storage.py"))
