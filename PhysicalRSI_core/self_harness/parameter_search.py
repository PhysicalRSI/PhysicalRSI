"""Bounded scalar refinement from verified development failures.

The task adapter classifies evidence as insufficient, satisfied or unresolved.
For each failed case, the search learns a bracket under an explicit monotonic
feasibility assumption. Rejected development probes remain useful observations.
It is numerical search, not an LLM, learned weights or permission to deploy.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
import json
import math
from pathlib import Path

from ..infra.storage import canonical, digest, file_digest, relative_path
from .artifacts import CLOSURE, harness_identity, verify_harness
from .search_space import materialize
from .evaluation import freeze_json
from .feedback_context import measurement_context


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Scalar search requires finite numbers')
    return float(value)


@dataclass(frozen=True)
class ScalarParameter:
    component: str
    file: str
    key: tuple[str, ...]
    lower: float
    upper: float
    precision: int = 3
    direction: str = 'increase'

    def __post_init__(self):
        if (self.component not in CLOSURE - {'foundation', 'weights'}
                or str(relative_path(self.file)) != self.file
                or not isinstance(self.key, (tuple, list)) or not self.key
                or any(not isinstance(k, str) or not k for k in self.key)
                or not _number(self.lower) < _number(self.upper)
                or type(self.precision) is not int or not 0 <= self.precision <= 9
                or self.direction not in {'increase', 'decrease'}):
            raise ValueError('Declare a bounded mutable scalar parameter')
        object.__setattr__(self, 'key', tuple(self.key))
        if any(round(v, self.precision) != v for v in (self.lower, self.upper)):
            raise ValueError('Parameter bounds must lie on the declared precision grid')

    def identity(self):
        return dict(component=self.component, file=self.file, key=list(self.key),
                    lower=self.lower, upper=self.upper, precision=self.precision, direction=self.direction)

    def inspect(self, candidate):
        verify_harness(candidate)
        owners = [kind for kind, files in candidate['components'].items() if self.file in files]
        if owners != [self.component]:
            raise ValueError('Scalar parameter requires one explicitly declared component owner')
        payload = (Path(candidate['root']) / self.file).read_bytes()
        if sha256(payload).hexdigest() != candidate['components'][self.component][self.file]:
            raise ValueError('Parameter document changed before parsing')
        document = json.loads(payload)
        node = document
        for key in self.key[:-1]:
            node = node[key]
        value = _number(node[self.key[-1]])
        if not self.lower <= value <= self.upper or round(value, self.precision) != value:
            raise ValueError('Candidate parameter is outside its declared search grid')
        fixed = deepcopy(document)
        self.assign(fixed, {'physicalrsi.scalar-slot/v1': True})
        components = deepcopy(candidate['components'])
        components[self.component][self.file] = digest(fixed)
        return dict(value=value, document=document, condition_sha256=digest(harness_identity(dict(candidate,components=components))))

    def assign(self, document, value):
        node = document
        for key in self.key[:-1]:
            node = node[key]
        if not isinstance(node, dict) or self.key[-1] not in node:
            raise ValueError('Scalar parameter key must already exist in a JSON object')
        node[self.key[-1]] = value


class ScalarRefinementEvolver:
    def __init__(self, *, archive, parameter, classifier):
        if not isinstance(parameter, ScalarParameter):
            raise ValueError('Declare a scalar search space')
        self.archive, self.parameter, self.classifier = archive, parameter, classifier
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind='development-scalar-refinement/v1', parameter=self.parameter.identity(),
            classifier=self.classifier.identity(), archive=self.archive.identity(),
            assumption='per-case monotone feasibility in the declared direction',
            implementation=file_digest(Path(__file__)),
            artifacts=file_digest(Path(__file__).with_name('artifacts.py')),
            feedback_context=file_digest(Path(__file__).with_name('feedback_context.py')),
            search_space=file_digest(Path(__file__).with_name('search_space.py')))

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError('Frozen scalar search implementation changed')

    def propose_from_archive(self, parent, evaluations, context, search, output):
        self._stable()
        parameter = self.parameter
        current = parameter.inspect(parent)
        records, excluded, evidence, panel = [], [], {}, {}

        def observe(candidate, value, source, extra=None, *, current_observation=False):
            origin = value['trial']['system2']
            if (origin is None or origin['split'] != 'development'
                    or origin['candidate_id'] != candidate['id']
                    or value['trial_sha256'] != digest(value['trial'])
                    or value['trial']['scope'] != search['view']['scope']
                    or origin['candidate_sha256'] != verify_harness(candidate)):
                raise ValueError('Scalar search requires attributed development Trials')
            inspected = parameter.inspect(candidate)
            case_id = digest(dict(task=value['trial']['task'], case=value['trial']['case'], scope=value['trial']['scope']))
            condition = measurement_context(value)
            sample = dict(source=source, trial_sha256=value['trial_sha256'], task=value['trial']['task'],
                          receipt_sha256=digest(value['receipt']), measurement_context=condition,
                          case=value['trial']['case'], case_sha256=case_id, value=inspected['value'])
            if current_observation:
                if case_id in panel:
                    raise ValueError('Current scalar development has duplicate cases')
                panel[case_id] = condition
            if inspected['condition_sha256'] != current['condition_sha256']:
                excluded.append(dict(sample, reason='other_controller_inputs_changed'))
                return
            if condition != panel[case_id]:
                excluded.append(dict(sample, reason='measurement_context_changed'))
                return
            outcome = value['outcome']
            classification = self.classifier.classify(deepcopy(value)) if outcome in {'success', 'failure'} else 'unresolved'
            if (classification not in {'satisfied', 'insufficient', 'unresolved'}
                    or classification == 'satisfied' and outcome != 'success'
                    or classification == 'insufficient' and outcome != 'failure'):
                raise ValueError('Scalar feedback classification contradicts the Trial outcome')
            records.append(dict(sample, classification=classification))
            for path, sha in chain(value['evidence'].items(), (extra or {}).items()):
                if path in evidence and evidence[path] != sha:
                    raise ValueError('Conflicting development evidence digests')
                evidence[path] = sha

        for value in evaluations:
            observe(parent, value, 'current_development', current_observation=True)
        if not records or len(evaluations) != len(records):
            raise ValueError('Current development must match the parameter context')
        targets = {row['case_sha256'] for row in records
                   if row['source'] == 'current_development' and row['classification'] == 'insufficient'}
        if targets:
            for row in self.archive.development(search['view']):
                value = row['evaluation']
                key = digest(dict(task=value['trial']['task'], case=value['trial']['case'], scope=value['trial']['scope']))
                if key in targets:
                    observe(row['candidate'], value, dict(archive_revision=row['archive_revision'], role=row['role']),
                            row['archive_evidence'])
        sign = 1 if parameter.direction == 'increase' else -1
        ceiling = parameter.upper if sign == 1 else -parameter.lower
        floor = sign * current['value']
        brackets, contradictions = {}, []
        for case_id in sorted(targets):
            failed = [sign*r['value'] for r in records if r['case_sha256'] == case_id and r['classification'] == 'insufficient']
            passed = [sign*r['value'] for r in records if r['case_sha256'] == case_id and r['classification'] == 'satisfied']
            low, high = max(failed), min(passed) if passed else ceiling
            if passed and low >= high:
                contradictions.append(case_id)
            brackets[case_id] = dict(lower_failed=low, upper_satisfied=high if passed else None, upper_bound=ceiling)
        candidate_value, reason = None, 'no_attributed_failure'
        if targets and not contradictions:
            low = max(floor, *(v['lower_failed'] for v in brackets.values()))
            high = max(v['upper_satisfied'] if v['upper_satisfied'] is not None else ceiling for v in brackets.values())
            proposed = round(sign * ((low + high) / 2), parameter.precision)
            if sign * proposed <= low:
                proposed = sign * high
            if low < sign * proposed <= ceiling:
                candidate_value, reason = proposed, 'refine_observed_failure_bound'
            else:
                reason = 'declared_bound_exhausted'
        elif contradictions:
            reason = 'contradictory_case_feedback'
        report = dict(schema='physicalrsi.scalar-refinement/v1', identity=self.identity(),
            parent_sha256=verify_harness(parent), condition_sha256=current['condition_sha256'],
            archive_view_sha256=digest(search['view']), measurement_contexts=panel, samples=records, excluded=excluded,
            brackets=brackets, contradictions=contradictions, proposed_value=candidate_value,
            reason=reason, qualification=None)
        output = Path(output)
        freeze_json(output / 'search-fit.json', report)
        self._stable()
        if candidate_value is None:
            return []
        document = deepcopy(current['document'])
        parameter.assign(document, candidate_value)
        payload = canonical(document) + b'\n'
        components = deepcopy(parent['components'])
        components[parameter.component][parameter.file] = sha256(payload).hexdigest()
        name = 'refinement-' + digest(report)[:12]
        destination = output / 'parameter-candidate'
        candidate = materialize(parent, components, {parameter.file: payload}, output=destination, name=name)
        self._stable()
        evidence[str(output / 'search-fit.json')] = file_digest(output / 'search-fit.json')
        return [dict(candidate, parent_sha256=verify_harness(parent), method='development-scalar-refinement',
            changes={parameter.component:dict(parameter=parameter.identity(), value=candidate_value)},
            environment=dict(scope=evaluations[0]['trial']['scope']), evidence=evidence, costs=dict(proposals=1),
            used_lessons=[], hypothesis='Refine the declared parameter past observed insufficient values on the failed development cases',
            expected_effect='Satisfy the task on independently evaluated cases under the explicit monotonicity assumption',
            required_checks=['fresh-paired-evaluation', 'complete-task-success'])]

    resume_proposal_from_archive = propose_from_archive
