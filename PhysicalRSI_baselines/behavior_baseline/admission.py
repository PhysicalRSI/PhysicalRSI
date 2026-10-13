"""Consume explicit code reviews and verify fixed native policy dependencies.

Reviews are trusted attestations, not an automatic code-safety proof. This gate
checks their candidate binding and physical source/checkpoint bytes. Environment
and simulator preflights remain scoped producer evidence, not a fresh simulator
qualification or a sandbox against malicious code.
"""
from copy import deepcopy
from pathlib import Path
import subprocess

from PhysicalRSI_core.infra.storage import digest, file_digest, read_json, relative_path
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.evaluation import freeze_json
from . import engine, openpi_backend

CHECKS = {'onboard_observations_only', 'native_controls_only', 'no_privileged_state_access',
          'no_evaluation_feedback_during_episode', 'no_runtime_program_rewriting'}


class ReviewedAdmission:
    def __init__(self, *, reference, reviews, evidence, tokenizer):
        self.reference = deepcopy(reference)
        self.reference_revision = verify_harness(reference)
        self.reviews = Path(reviews).resolve()
        self.evidence = {str(Path(p).resolve()): sha for p, sha in evidence.items()}
        self.tokenizer = Path(tokenizer).resolve()
        if not self.evidence:
            raise ValueError('Bind actual environment and native preflight evidence')
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind='behavior-reviewed-admission/v1', reference=self.reference_revision,
            reviews=str(self.reviews), evidence=self.evidence, tokenizer=str(self.tokenizer),
            checks=sorted(CHECKS), implementations={p.name: file_digest(p) for p in
                (Path(__file__), Path(engine.__file__), Path(openpi_backend.__file__))})

    def check(self, candidate, directory):
        if self.identity() != self._identity or verify_harness(self.reference) != self.reference_revision:
            raise ValueError('Admission policy or reference changed')
        revision = verify_harness(candidate)
        if any(candidate['components'][key] != self.reference['components'][key]
               for key in CLOSURE - {'control'}):
            raise ValueError('Only reviewed control changes are admitted under this fixed foundation')
        engine.verify(candidate)
        review_path = self.reviews / (revision + '.json')
        review = read_json(review_path)
        if (review.get('schema') != 'physicalrsi.behavior-code-review/v1'
                or review.get('candidate_sha256') != revision
                or review.get('control_sha256') != candidate['components']['control'].get('control.py')
                or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or not isinstance(review.get('reason'), str) or not review['reason'].strip()
                or type(review.get('accepted')) is not bool
                or set(review.get('checks', {})) != CHECKS
                or any(type(value) is not bool for value in review['checks'].values())):
            raise ValueError('Require an explicit complete review for this exact program')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        freeze_json(directory / 'review.json', review)
        accepted = review['accepted'] and all(review['checks'].values())
        bound = {}
        if accepted:
            for path, sha in self.evidence.items():
                if file_digest(Path(path)) != sha:
                    raise ValueError('Admission prerequisite evidence changed')
                bound[path] = sha
            backend = read_json(Path(candidate['root']) / 'backend.json')
            if (backend.get('kind') != 'openpi-b1k/v1'
                    or backend.get('source_revision') != openpi_backend.OPENPI_REVISION
                    or backend.get('adapter_sha256') != file_digest(Path(openpi_backend.__file__))
                    or backend.get('tokenizer') != openpi_backend.TOKENIZER):
                raise ValueError('Unexpected fixed official action backend')
            source = Path(backend['source_root']).resolve()
            actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                             text=True, timeout=30).strip()
            if actual != backend['source_revision']:
                raise ValueError('Official backend source revision changed')
            subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--'],
                           check=True, timeout=120)
            if file_digest(self.tokenizer) != backend['tokenizer']['sha256']:
                raise ValueError('Pinned tokenizer bytes changed')
            root = Path(backend['checkpoint_root']).resolve()
            files = backend['checkpoint_files']
            actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
            if not files or actual_files != set(files):
                raise ValueError('Checkpoint inventory changed')
            for name, sha in files.items():
                path = root / relative_path(name)
                if (path.is_symlink() or not path.resolve().is_relative_to(root)
                        or file_digest(path) != sha):
                    raise ValueError('Checkpoint bytes changed')
            if verify_harness(candidate) != revision or self.identity() != self._identity:
                raise ValueError('Candidate or admission implementation changed during checks')
            # Preserve prerequisite bytes as well as references for core lineage.
            for index, (path, sha) in enumerate(sorted(bound.items())):
                target = directory / f'prerequisite-{index:03d}.bin'
                data = Path(path).read_bytes()
                if target.exists() and target.read_bytes() != data:
                    raise ValueError('Frozen prerequisite copy changed')
                if not target.exists():
                    with target.open('xb') as stream:
                        stream.write(data)
                if file_digest(target) != sha:
                    raise ValueError('Prerequisite evidence changed while copying')
        report = dict(candidate_sha256=revision, accepted=accepted,
            reason=review['reason'], review_sha256=digest(review), prerequisites=bound,
            checked_checkpoint_inventory=accepted, qualification=None,
            scope='Reviewed control eligibility and fixed source/weight integrity; no task-performance qualification')
        freeze_json(directory / 'checks.json', report)
        names = ['review.json', 'checks.json'] + [f'prerequisite-{i:03d}.bin' for i in range(len(bound))]
        return dict(accepted=accepted, reason=review['reason'],
                    evidence={name: file_digest(directory / name) for name in names})
