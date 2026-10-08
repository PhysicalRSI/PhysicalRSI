"""Core Self-Harness evaluator port for candidate-bound RobotWorld episodes.

The port deliberately keeps task sampling and scoring in the pinned native
worker. It only translates validated receipts into the core paired-selection
schema; it never turns a launch failure into a score.
"""
from datetime import datetime, timezone
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest
from PhysicalRSI_core.self_harness.artifacts import verify_harness

from .native_trials import RoboCasaTrials
from .policy_bundle import read_bundle


class RoboCasaNativeEvaluator:
    def __init__(self, *, trials, fixed_components, task_indices, budgets,
                 launcher_seed=7):
        self.trials = trials
        self.fixed_components = fixed_components
        self.task_indices = dict(task_indices)
        self.budgets = dict(budgets)
        self.launcher_seed = launcher_seed
        if type(launcher_seed) is not int or launcher_seed < 0:
            raise ValueError('Launcher seed must be a nonnegative integer')
        self.revision = digest(self.identity())

    def identity(self):
        return dict(kind='robotworld-robocasa-native-evaluator/v1',
                    trials=self.trials.identity(), task_indices=self.task_indices,
                    budgets=self.budgets, launcher_seed=self.launcher_seed,
                    fixed_components=self.fixed_components)

    def admit(self, candidate, output):
        sha = verify_harness(candidate)
        read_bundle(candidate, fixed=self.fixed_components)
        path = Path(output) / ('admission-' + candidate['id'] + '.json')
        atomic_json(path, dict(candidate_id=candidate['id'], freeze_sha256=sha,
                              accepted=True, scope='robotworld_native_task_adaptation'))
        return dict(accepted=True, freeze_sha256=sha,
                    evidence={str(path.relative_to(Path(output))): file_digest(path)})

    def _cases(self, comparison):
        cases = {}
        for task, spec in comparison['profile']['tasks'].items():
            if task not in self.task_indices or task not in self.budgets:
                raise ValueError('Missing pinned RoboCasa task index or budget')
            values = []
            for slot in range(spec['episodes']):
                launcher = self.launcher_seed + slot
                values.append(dict(task=task, launcher_seed=launcher,
                                   reset_seed=launcher + self.task_indices[task],
                                   task_index=self.task_indices[task],
                                   horizon=self.budgets[task], task_set='all_tasks',
                                   split='pretrain', trial_slot=slot))
            cases[task] = values
        return cases

    def validation(self, comparison, output):
        cases = self._cases(comparison)
        layouts = {task: [digest(case) for case in values] for task, values in cases.items()}
        path = Path(output) / 'native-validation-cases.json'
        atomic_json(path, cases)
        return dict(split='validation', comparison_sha256=digest(comparison),
                    generated_at=datetime.now(timezone.utc).isoformat(), layouts=layouts,
                    admission_evidence={str(path.relative_to(Path(output))): file_digest(path)})

    def evaluate(self, candidate, comparison, cohort, output):
        sha = verify_harness(candidate)
        if comparison['candidates'].get(candidate['id']) != sha:
            raise ValueError('Candidate is absent from the frozen comparison')
        cases = self._cases(comparison)
        root = Path(output).resolve()
        episodes = []
        for task, values in cases.items():
            for case in values:
                run_id = task + '-slot-' + str(case['trial_slot'])
                trial_root = root / 'native-trials' / candidate['id'] / run_id
                result = self.trials.run(candidate, case, trial_root, seconds=7200)
                evidence = {str((trial_root / name).relative_to(root)): value
                            for name, value in result['evidence'].items()}
                episodes.append(dict(task=task, layout_sha256=digest(case), state='completed',
                                     success=result['outcome'] == 'success', score=result['score'],
                                     evidence_sha256=evidence))
        return dict(kind='policy_evaluation', candidate_id=candidate['id'],
                    freeze_sha256=sha, comparison_sha256=digest(comparison),
                    cohort_sha256=digest(cohort), evaluator_revision=self.revision,
                    native_exit_code=0, experiment_protocol_sha256=self.revision,
                    episodes=episodes)
