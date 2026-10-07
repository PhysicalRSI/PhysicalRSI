import pytest
from PhysicalRSI_core.infra.storage import atomic_json, file_digest
from PhysicalRSI_baselines.embodied_goodharts_law.public_diagnostics import export_trial


def fixture(tmp_path, reply=None):
    trajectory = tmp_path / 'trajectory.json'
    atomic_json(trajectory, [{'step': 0, 'action': {'kind': 'call', 'method': 'locate_object'},
                            'observation': {'reply': reply or {'id': 0, 'value': {'status': 'not_detected'}},
                                            'evaluation': {'object_poses': 'private', 'official_success': False}}}])
    receipt = tmp_path / 'receipt.json'
    atomic_json(receipt, {'state': 'completed', 'outcome': 'failure',
                         'evidence': {'trajectory.json': file_digest(trajectory)},
                         'verdict': {'measurements': {'bottle_model_ids': [19]}}})
    return receipt, file_digest(receipt)


def test_exports_public_feedback_and_score_without_evaluator_truth(tmp_path):
    receipt, sha = fixture(tmp_path)
    value = export_trial(receipt, expected_receipt_sha256=sha)
    assert value['official_outcome'] == 'failure'
    assert value['trajectory'] == [{'step': 0, 'action': {'kind': 'call', 'method': 'locate_object'},
                                    'reply': {'id': 0, 'value': {'status': 'not_detected'}}}]
    assert not value['boundary']['prior_gt_assisted_development_erased']


@pytest.mark.parametrize('field', ['object_poses', 'actor_segmentation', 'official_success'])
def test_rejects_nested_privileged_reply(tmp_path, field):
    receipt, sha = fixture(tmp_path, {'id': 0, 'value': {'nested': [{field: 'private'}]}})
    with pytest.raises(ValueError, match='Privileged'):
        export_trial(receipt, expected_receipt_sha256=sha)


@pytest.mark.parametrize('filename', ['receipt.json', 'trajectory.json'])
def test_rejects_evidence_changed_after_recording(tmp_path, filename):
    receipt, sha = fixture(tmp_path)
    (tmp_path / filename).write_text('{}')
    with pytest.raises(ValueError, match='digest differs'):
        export_trial(receipt, expected_receipt_sha256=sha)
