"""Freeze editable System 1 inputs without giving candidates evaluator control.

These files are data on the host. Python skills are parsed, never executed here;
the native worker must expose the materialized directory read-only at /policy
inside the isolated agent. This module alone does not provide that mount.
"""
import ast
from copy import deepcopy
from pathlib import Path

from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.proposals import JsonEditContract, ProposalLimits


def edit_contract():
    def object_schema(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    return JsonEditContract({
        'prompts.json': dict(component='prompts', schema=object_schema({
            'guidance': dict(type='string', maxLength=16384)})),
        'skills.json': dict(component='skills', schema=object_schema({
            'files': dict(type='object', maxProperties=16, additionalProperties=False,
                          patternProperties={r'^[a-z][a-z0-9_]{0,63}\.py$':
                                             dict(type='string', maxLength=65536)})})),
        'memory.json': dict(component='memory_rules', schema=object_schema({
            'lessons': dict(type='array', maxItems=64, items=dict(type='string', maxLength=2048))})),
    })


def _fixed(candidate):
    editable = set(edit_contract().files)
    return {kind: {name: sha for name, sha in entries.items() if name not in editable}
            for kind, entries in candidate['components'].items()}


def fixed_components(reference):
    """Build the trusted deployment's reference, not a candidate self-attestation."""
    edit_contract().inputs(reference, ProposalLimits())
    return deepcopy(_fixed(reference))


def read_bundle(candidate, *, fixed):
    """Reject changes outside the deployment's reviewed editable file domain."""
    inputs = edit_contract().inputs(candidate, ProposalLimits())
    if _fixed(candidate) != fixed:
        raise ValueError('Candidate changed fixed runtime, tools, foundation or task inputs')
    bundle = {name: deepcopy(row['value']) for name, row in inputs.items()}
    for name, source in bundle['skills.json']['files'].items():
        ast.parse(source, filename=name)
    return bundle


def materialize(candidate, *, fixed, destination):
    """Prepare a fresh read-only agent mount and immutable initial memory revision.

    The host worker owns these bytes. The agent may create working notes in its
    separate writable workspace; promoting them requires a new candidate.
    """
    sha = verify_harness(candidate)
    bundle = read_bundle(candidate, fixed=fixed)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    if root.is_symlink():
        raise ValueError('Policy inputs need a physical directory')
    snapshot = MemoryStore(root / 'memory').snapshot(bundle['memory.json'])
    skills = []
    for name, source in bundle['skills.json']['files'].items():
        path = root / 'skills' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
        skills.append(path.relative_to(root).as_posix())
    atomic_json(root / 'context.json', dict(candidate_sha256=sha,
        guidance=bundle['prompts.json']['guidance'], skills=sorted(skills),
        memory='memory/' + snapshot.revision + '.json', memory_revision=snapshot.revision))
    if verify_harness(candidate) != sha:
        raise ValueError('Candidate changed during materialization')
    return dict(scope='system1_input_bundle', qualification=None, candidate_sha256=sha,
                memory_revision=snapshot.revision,
                files={p.relative_to(root).as_posix(): file_digest(p)
                       for p in root.rglob('*') if p.is_file() and p.name != '.write.lock'})
