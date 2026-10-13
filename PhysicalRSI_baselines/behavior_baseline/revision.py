"""Materialize a reviewed System 2 control revision without changing its foundation.

This never executes proposal code or admits/selects a candidate. The caller owns
the scientific hypothesis, code review and independent evaluation. Feedback must
come from the development exporter; hashes do not certify its upstream admission.
"""
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path
from PhysicalRSI_core.self_harness.artifacts import verify_harness


def materialize(output, *, parent, control, feedback, feedback_root, hypothesis,
                environment, costs, expected_effect, required_checks, used_lessons=()):
    parent = deepcopy(parent)
    parent_sha = verify_harness(parent)
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("Record the development hypothesis before materialization")
    if any(not isinstance(value, dict) or not value for value in (environment, costs)):
        raise ValueError("Declare proposal environment and measured costs")
    if not isinstance(expected_effect, str) or not expected_effect.strip():
        raise ValueError("Declare the expected effect before evaluation")
    if (not isinstance(required_checks, list) or not required_checks
            or any(not isinstance(value, str) or not value.strip() for value in required_checks)):
        raise ValueError("Declare nonempty required evaluation checks")
    if (not isinstance(used_lessons, (list, tuple))
            or any(not isinstance(value, str) or not value for value in used_lessons)
            or len(set(used_lessons)) != len(used_lessons)):
        raise ValueError("Used lessons must be unique research memory identifiers")
    provenance = deepcopy(dict(environment=environment, costs=costs,
        hypothesis=hypothesis, expected_effect=expected_effect,
        required_checks=required_checks, used_lessons=list(used_lessons)))
    digest(provenance)  # Reject non-JSON/nonfinite declarations before writing.
    if set(parent['components']['control']) != {'control.py'}:
        raise ValueError("Expected one declared control.py program")
    if any('control.py' in entries for name, entries in parent['components'].items() if name != 'control'):
        raise ValueError("Control must not be shared with a fixed foundation component")
    declared_files = {name for entries in parent['components'].values() for name in entries}
    if declared_files & {'candidate.json', 'proposal.json'}:
        raise ValueError("Candidate and proposal metadata must remain outside the executable closure")
    feedback = deepcopy(feedback)
    feedback_root = Path(feedback_root).resolve()
    if feedback.get('split') != 'evolve' or not feedback.get('evidence'):
        raise ValueError("Use exported development feedback, never validation or test feedback")

    def verify_feedback():
        for name, sha in feedback['evidence'].items():
            path = (feedback_root / relative_path(name)).resolve()
            if not path.is_relative_to(feedback_root) or file_digest(path) != sha:
                raise ValueError("Development feedback evidence changed")

    verify_feedback()
    summary_name = 'development-feedback/episodes.json'
    if summary_name not in feedback['evidence']:
        raise ValueError("Missing bound development summary")
    summary = read_json(feedback_root / summary_name)
    if (summary.get('schema') != 'physicalrsi.behavior-development-feedback/v1'
            or summary.get('parent_id') != parent['id']
            or summary.get('parent_revision') != parent_sha
            or not summary.get('episodes')
            or summary['episodes'] != feedback.get('episodes')
            or summary.get('evaluator_revision') != feedback.get('evaluator_revision')):
        raise ValueError("Development feedback belongs to another parent or evaluator")
    code = Path(control).read_bytes()
    syntax = ast.parse(code, filename='control.py')
    names = {node.name for node in syntax.body if isinstance(node, ast.FunctionDef)}
    if not {'initialize', 'step'} <= names:
        raise ValueError("Control must define initialize and step")
    source = Path(parent['root']).resolve()
    if code == (source / 'control.py').read_bytes():
        raise ValueError("A revision must change the control program")
    output = Path(output).absolute()
    identifier(output.name)
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Keep new revisions outside the immutable parent directory")
    if output.name == parent['id']:
        raise ValueError("Use a distinct candidate ID")
    output.parent.mkdir(parents=True, exist_ok=True)
    with locked(output.parent / ('.' + output.name + '.lock')):
        if output.exists() or output.is_symlink():
            raise FileExistsError("Never overwrite a candidate revision")
        stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
        try:
            for name in sorted(declared_files):
                destination = stage / relative_path(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)
            (stage / 'control.py').write_bytes(code)
            child = dict(id=output.name, root=str(stage), components=deepcopy(parent['components']),
                parent_sha256=parent_sha, method='reviewed-control-revision',
                changes={'control.py': hypothesis},
                evidence=dict(root=str(feedback_root), files=deepcopy(feedback['evidence']),
                              feedback_sha256=digest(feedback)), **provenance)
            child['components']['control'] = {'control.py': file_digest(stage / 'control.py')}
            child_sha = verify_harness(child)
            if verify_harness(parent) != parent_sha:
                raise ValueError("Parent changed during materialization")
            verify_feedback()
            child['root'] = str(output)
            atomic_json(stage / 'candidate.json', child)
            atomic_json(stage / 'proposal.json', dict(
                schema='physicalrsi.behavior-control-revision/v1', parent_id=parent['id'],
                parent_sha256=parent_sha, candidate_sha256=child_sha,
                feedback_root=str(feedback_root), feedback=feedback,
                feedback_sha256=digest(feedback), hypothesis=hypothesis,
                provenance=provenance,
                implementation_sha256=file_digest(Path(__file__)), qualification=None,
                scope='Reviewed control materialization only; admission and evaluation pending'))
            stage.rename(output)
            return child
        finally:
            if stage.exists():
                shutil.rmtree(stage)
