"""RobotWorld campaigns using the shared Self-Harness and CAS lineage.

Proposer and evaluator ports own Codex execution, isolation, native launches and
independent evidence. Constructing this campaign does not supply those ports or
qualify a simulator. Their executable identities are frozen by SelfHarness.
"""
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, file_digest
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness, selection
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign


class NativeSelector:
    def identity(self):
        return {'implementation': file_digest(Path(selection.__file__))}

    select = staticmethod(selection.select_survivor)


def campaign(workspace, *, catalog, benchmark, tasks, proposer, evaluator,
             rounds, episodes=3, minimum_gain=0.0):
    """Build a bounded adaptation campaign, retaining the original task budgets.

    The evaluator must admit the frozen candidate, bind each native rollout to
    it, verify isolation and independently score paired cases. RoboCasa result
    parsing is available in receipts.py; parsing alone is not admission.
    Repeated fixed-instance trials must not be described as unseen layouts.
    """
    if digest(catalog['body']) != catalog['revision']:
        raise ValueError('Catalog identity changed')
    body = catalog['body']
    if body.get('schema') != 'physicalrsi.robotworld-catalog/v1' or body.get('code_control') is not False:
        raise ValueError('Expected the reviewed RobotWorld default-control protocol')
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Declare unique tasks for this benchmark')
    inventory = {row['task']: row for row in body['tasks'] if row['benchmark'] == benchmark}
    if not set(tasks) <= set(inventory):
        raise ValueError('Task not in the pinned benchmark inventory')
    selected = {task: inventory[task] for task in tasks}
    profile = dict(tasks={task: dict(weight=1, episodes=episodes, score_range=[0, 1],
                                     maximum_regression=0) for task in tasks},
                   minimum_gain=minimum_gain, tie_tolerance=0)
    selection.validate_profile(profile)
    protocol = dict(evaluation_kind='policy_evaluation', catalog_revision=catalog['revision'],
                    upstream_revision=body['upstream_revision'], benchmark=benchmark,
                    tasks=selected, code_control=False, metric='native_binary_success',
                    test_role='report_only', trial_role='paired_adaptation_validation',
                    reset_novelty='not_implied_by_repeated_rollouts')
    root = Path(workspace).resolve()
    state = HarnessState(root / 'state')
    def build_loop(directory):
        return SelfHarness(directory, state=state, proposer=proposer, evaluator=evaluator,
                           selector=NativeSelector(), profile=profile, protocol=protocol,
                           scope='robotworld_native_task_adaptation', max_candidates=3)
    return ImprovementCampaign(root / 'campaign', state=state, build_loop=build_loop,
                               max_rounds=rounds)
