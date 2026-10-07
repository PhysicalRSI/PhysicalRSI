"""Bind three editable EGL artifacts to the existing isolated CAP executor.

This binds declared bytes; it does not attest dependency closure, admit a
benchmark trial, call a model, or select a survivor. Candidate Python is parsed
on the host and executed only when the resulting program enters isolation.
"""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from PhysicalRSI_core.infra.storage import canonical, digest, file_digest, strict_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.proposals import JsonEditContract


PRIMITIVES = "egl/primitive_skills.json"
COMBINATIONS = "egl/skill_combinations.json"
MEMORY = "egl/memory.json"
MAX_ARTIFACT_BYTES = 256 * 1024


def _program_schema(kind):
    return dict(type="object", properties={
        "schema": {"const": kind},
        "source": dict(type="string", minLength=1, maxLength=100000)},
        required=["schema", "source"], additionalProperties=False)


def edit_contract():
    """Three independently editable files in Self-Harness's existing closure."""
    return JsonEditContract({
        PRIMITIVES: dict(component="skills", schema=_program_schema("physicalrsi.egl-primitives/v1")),
        COMBINATIONS: dict(component="skill_selection", schema=_program_schema("physicalrsi.egl-combinations/v1")),
        MEMORY: dict(component="memory_rules", schema=dict(
            type="object", properties={"schema": {"const": "physicalrsi.egl-memory/v1"},
                                       "memory": dict(type="object")},
            required=["schema", "memory"], additionalProperties=False)),
    })


@dataclass(frozen=True)
class BoundCandidateProgram:
    source: str
    memory_json: str
    harness_sha256: str
    artifact_digests: tuple
    binding_implementation: str

    @property
    def memory(self):
        # Every episode receives a fresh copy; runtime mutations are not inherited.
        return json.loads(self.memory_json)

    def identity(self):
        return dict(kind="egl-isolated-candidate-program/v1",
                    harness_sha256=self.harness_sha256,
                    artifacts=dict(self.artifact_digests),
                    source_sha256=digest(self.source), memory_sha256=digest(self.memory),
                    binding_implementation=self.binding_implementation)


def bind_candidate(candidate):
    """Load a verified snapshot without importing or executing proposed Python.

    The primitive library exports names through ``skills``. The combination
    defines ``policy(robot, memory)`` and may call those functions. The robot
    capabilities and execution limits remain outside this editable domain.
    """
    revision = verify_harness(candidate)
    root = Path(candidate["root"])
    rules = edit_contract().public()
    values, hashes = {}, {}
    for name, rule in rules.items():
        owners = [kind for kind, entries in candidate["components"].items() if name in entries]
        if owners != [rule["component"]]:
            raise ValueError("EGL artifact must belong exclusively to its declared component")
        with (root / name).open("rb") as stream:
            payload = stream.read(MAX_ARTIFACT_BYTES + 1)
        if len(payload) > MAX_ARTIFACT_BYTES:
            raise ValueError("EGL artifact exceeds binding byte allowance")
        expected = candidate["components"][rule["component"]][name]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("EGL artifact changed while being read")
        value = strict_json(payload)
        if not Draft202012Validator(rule["schema"]).is_valid(value):
            raise ValueError("EGL artifact does not match its declared schema")
        values[name], hashes[name] = value, expected
    ast.parse(values[PRIMITIVES]["source"], filename=PRIMITIVES)
    tree = ast.parse(values[COMBINATIONS]["source"], filename=COMBINATIONS)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "policy" for node in tree.body):
        raise ValueError("Skill combinations must define policy(robot, memory)")
    # These exec calls are emitted as data. The host must pass this source to
    # execute_policy (or another admitted isolated executor), never exec it.
    source = ("_egl_skills = {}\n"
              + "exec(compile(" + repr(values[PRIMITIVES]["source"]) + ", '<primitive-skills>', 'exec'), _egl_skills)\n"
              + "_egl_combinations = {'skills': _egl_skills}\n"
              + "exec(compile(" + repr(values[COMBINATIONS]["source"]) + ", '<skill-combinations>', 'exec'), _egl_combinations)\n"
              + "def policy(robot, memory):\n"
              + "    return _egl_combinations['policy'](robot, memory)\n")
    if verify_harness(candidate) != revision:
        raise ValueError("Candidate changed while binding")
    return BoundCandidateProgram(source, canonical(values[MEMORY]["memory"]).decode(),
                                 revision, tuple(sorted(hashes.items())), file_digest(Path(__file__)))
