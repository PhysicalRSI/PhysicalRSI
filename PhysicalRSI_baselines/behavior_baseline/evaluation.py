"""Native BEHAVIOR evaluation ports for the existing core Self-Harness.

Admission is an explicit reviewed producer port, not inferred from hashes or
software tests. Its check(candidate, directory) must return accepted, reason
(on rejection), and relative evidence-file hashes, without simulator effects.
The batch backend requires explicit OpenPI checkpoint task coverage; unsupported
coverage fails before execution rather than silently reducing the task scope.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from PhysicalRSI_core.embodiment import Revision, System2Trial
from PhysicalRSI_core.infra.storage import digest, file_digest, identifier, locked, read_json, relative_path
from PhysicalRSI_core.self_harness import now
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.bindings import validate_admission
from PhysicalRSI_core.self_harness.evaluation import freeze_json
from PhysicalRSI_core.self_harness.protocol import CaseProtocol
from PhysicalRSI_core.self_harness.selection import comparison_identity
from . import batch, bridge, feedback, grouped, native, service, trials
from .openpi_backend import supported_tasks


class NativeEvaluator:
    def __init__(self, runtime, *, protocol, admission, quota, python, system2,
                 scope, port=8000, seconds=3600, startup_seconds=1800,
                 temporary_directory=None, native_group_size=1):
        if not isinstance(protocol, CaseProtocol) or not isinstance(system2, Revision) or not scope:
            raise ValueError('Declare a case protocol, System 2 identity and scope')
        self.runtime = Path(runtime).resolve()
        self.protocol, self.admission, self.quota = protocol, admission, quota
        self.python = Path(python).absolute()
        self.system2, self.scope = system2, scope
        self.options = dict(port=port, seconds=seconds, temporary_directory=temporary_directory)
        self.startup_seconds = startup_seconds
        if type(native_group_size) is not int or native_group_size < 1:
            raise ValueError('Declare a positive integer native group size')
        self.native_group_size = native_group_size
        self._identity = deepcopy(self.identity())
        self.revision = digest(self._identity)

    def identity(self):
        return dict(kind='behavior-native-evaluator/v1', runtime=str(self.runtime),
            protocol=self.protocol.identity(), admission=self.admission.identity(),
            quota=self.quota.identity(), python=str(self.python), system2=asdict(self.system2),
            scope=self.scope, options=self.options, startup_seconds=self.startup_seconds,
            native_group_size=self.native_group_size,
            implementations={p.name: file_digest(p) for p in [Path(__file__),
                *[Path(m.__file__) for m in (batch, bridge, feedback, grouped, native, service, trials)]]})

    def policy_identity(self):
        value = self.identity()
        value['protocol'] = self.protocol.policy_identity()
        return value

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError('Native evaluator identity changed')

    def register_round(self, state, round_identity, *, context=None):
        self._stable()
        state.reserve_splits(round_identity, self.protocol.reset_identities())

    def _output(self, output):
        output = Path(output).resolve()
        if self.quota.root.is_relative_to(output):
            raise ValueError('Keep mutable quota outside round evidence')
        return output

    def _admitted(self, output, name, revision):
        directory = output / 'native-admission' / identifier(name)
        record = read_json(directory / 'record.json')
        candidate = read_json(directory / 'candidate.json')
        receipt = record['receipt']
        validate_admission(receipt, revision)
        if (record['evaluator_revision'] != self.revision or candidate['id'] != name
                or verify_harness(candidate) != revision):
            raise ValueError('Admission candidate or evaluator changed')
        for relative, sha in receipt['evidence'].items():
            path = (directory / relative_path(relative)).resolve()
            if not path.is_relative_to(directory) or file_digest(path) != sha:
                raise ValueError('Admission evidence changed')
        return candidate, receipt, directory

    def admit(self, candidate, output):
        self._stable()
        output = self._output(output)
        revision = verify_harness(candidate)
        directory = output / 'native-admission' / identifier(candidate['id'])
        with locked(directory / '.lock'):
            if not (directory / 'record.json').exists():
                receipt = deepcopy(self.admission.check(deepcopy(candidate), directory))
                receipt['freeze_sha256'] = revision
                validate_admission(receipt, revision)
                self._stable()
                if verify_harness(candidate) != revision:
                    raise ValueError('Candidate changed during admission')
                freeze_json(directory / 'candidate.json', candidate)
                freeze_json(directory / 'record.json', dict(evaluator_revision=self.revision, receipt=receipt))
            _, receipt, directory = self._admitted(output, candidate['id'], revision)
        return dict(receipt, evidence={str((directory / name).relative_to(output)): sha
                                     for name, sha in receipt['evidence'].items()})

    def _cases(self, split, comparison=None):
        cases = (self.protocol.claim_validation(comparison) if comparison is not None
                 else self.protocol.cases(split))
        fields = {'task', 'mode', 'instance_id', 'instance_index', 'rollout_id'}
        for task, values in cases.items():
            for case in values:
                if (set(case) != fields or case['task'] != task or case['mode'] != 'train'
                        or type(case['instance_id']) is not int or case['instance_id'] < 0
                        or case['instance_index'] != case['instance_id'] or case['rollout_id'] != 0):
                    raise ValueError('Use canonical native train cases from RoundCases')
        if comparison is not None:
            profile = comparison['profile']['tasks']
            if (set(profile) != set(cases) or any(len(cases[t]) != p['episodes']
                    or p['weight'] != 1 or p['score_range'] != [0, 1] for t, p in profile.items())):
                raise ValueError('Require complete equal-weight native Q-score task coverage')
        return cases

    def _plan(self, output, candidates, cases, split, comparison=None, cohort=None):
        requests = []
        for name, revision in sorted(candidates.items()):
            candidate, receipt, _ = self._admitted(output, name, revision)
            if not receipt['accepted']:
                raise ValueError('Candidate did not pass independent admission')
            backend = read_json(Path(candidate['root']) / 'backend.json')
            if not set(cases) <= supported_tasks(backend):
                raise ValueError('Checkpoint backend does not cover the entire declared task set')
            origin = System2Trial(self.system2, name, revision, split,
                                  digest(comparison) if comparison else None,
                                  digest(cohort) if cohort else None)
            for task, values in sorted(cases.items()):
                requests.extend(dict(task=task, mode='train', instance_index=c['instance_index'],
                                     system2=origin, **self.options) for c in values)
        if self.native_group_size > 1 and split != 'development':
            return grouped.GroupedNativeTrials(self.runtime, output / 'native-trials' / split,
                quota=self.quota, trials=requests, group_size=self.native_group_size)
        return trials.NativeTrials(self.runtime, output / 'native-trials' / split,
                                   quota=self.quota, trials=requests)

    def _execute(self, plan, output, candidates):
        self._stable()
        registry = {}
        for name, revision in candidates.items():
            _, receipt, directory = self._admitted(output, name, revision)
            if not receipt['accepted']:
                raise ValueError('Candidate admission was rejected')
            registry[name] = directory / 'candidate.json'
        batch.execute(plan, candidates=registry, python=self.python,
                      startup_seconds=self.startup_seconds, episode_seconds=self.options['seconds'])
        self._stable()

    def development(self, parent, output):
        output = self._output(output)
        if not self.admit(parent, output)['accepted']:
            raise ValueError('Incumbent failed admission before development')
        revision = verify_harness(parent)
        cases = self._cases('development')
        plan = self._plan(output, {parent['id']: revision}, cases, 'development')
        self._execute(plan, output, {parent['id']: revision})
        rows = [dict(folder=plan.output / 'trials' / r['id'], specification=r['specification'])
                for r in plan.plan['trials']]
        return feedback.export_development(rows, parent_id=parent['id'], parent_revision=revision,
            expected_cases=[c for values in cases.values() for c in values],
            evaluator_revision=self.revision, output=output)

    def validation(self, comparison, output):
        self._stable()
        output = self._output(output)
        comparison_sha = comparison_identity(comparison)
        if comparison.get('scope') != self.scope:
            raise ValueError('Comparison scope differs from evaluator scope')
        with locked(output / '.native-validation.lock'):
            cases = self._cases('validation', comparison)
            path = output / 'native-validation-cases.json'
            freeze_json(path, cases)
            cohort_path = output / 'native-validation-cohort.json'
            generated = read_json(cohort_path)['generated_at'] if cohort_path.exists() else now()
            cohort = dict(split='validation', comparison_sha256=comparison_sha, generated_at=generated,
                layouts={t: [digest(c) for c in values] for t, values in cases.items()},
                admission_evidence={path.name: file_digest(path)})
            freeze_json(cohort_path, cohort)
            self._plan(output, comparison['candidates'], cases, 'validation', comparison, cohort)
            return cohort

    def evaluate(self, candidate, comparison, cohort, output):
        output = self._output(output)
        if comparison['candidates'].get(candidate['id']) != verify_harness(candidate):
            raise ValueError('Candidate differs from frozen comparison')
        if self.validation(comparison, output) != cohort:
            raise ValueError('Validation cohort changed')
        plan = self._plan(output, comparison['candidates'], self._cases('validation', comparison),
                          'validation', comparison, cohort)
        # The complete pool is reserved before its first effect. Subsequent
        # core evaluate calls reuse verified results without rerunning trials.
        self._execute(plan, output, comparison['candidates'])
        rows = [dict(folder=plan.output / 'trials' / r['id'], specification=r['specification'])
                for r in plan.plan['trials']
                if r['inputs']['system2']['candidate_id'] == candidate['id']]
        return bridge.selection_result(rows, candidate_id=candidate['id'], comparison=comparison,
            cohort=cohort, evaluator_revision=self.revision, evidence_root=output)

    resume_validation = validation
    resume_evaluation = evaluate
