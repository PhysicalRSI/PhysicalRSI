"""Export verified public CAP feedback for subsequent System 2 diagnosis.

This does not erase prior GT-assisted development or attest the perception
service. Simulator-rendered depth remains part of the declared sensor input.
"""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest, read_json

PRIVATE_FIELDS = frozenset({
    'evaluation', 'official_success', 'object_displacements',
    'post_settle_object_displacements', 'bottle_model_ids', 'model_ids',
    'reset_geometry_sha256', 'obj_body_id', 'actor_segmentation',
    'mesh_segmentation', 'object_pose', 'object_poses', 'actor_pose', 'actor_poses',
})


def _check_public(value):
    if isinstance(value, dict):
        if PRIVATE_FIELDS.intersection(value):
            raise ValueError('Privileged field in purported public CAP feedback')
        for item in value.values():
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)


def export_trial(receipt_path, *, expected_receipt_sha256):
    """Keep requests/replies and the official outcome, withholding measurements.

    The expected receipt digest must come from independently retained evidence.
    Field checks supplement source review; they cannot detect encoded leaks.
    """
    receipt_path = Path(receipt_path)
    if file_digest(receipt_path) != expected_receipt_sha256:
        raise ValueError('Receipt digest differs from retained evidence')
    receipt = read_json(receipt_path)
    if receipt.get('state') != 'completed':
        raise ValueError('Only completed trials may enter diagnosis')
    trajectory_path = receipt_path.parent / 'trajectory.json'
    trajectory_sha = receipt['evidence']['trajectory.json']
    if file_digest(trajectory_path) != trajectory_sha:
        raise ValueError('Trajectory digest differs from receipt')
    public = []
    for row in read_json(trajectory_path):
        entry = {key: row[key] for key in ('step', 'action') if key in row}
        reply = row.get('observation', {}).get('reply')
        if reply is not None:
            entry['reply'] = reply
        _check_public(entry)
        if entry:
            public.append(entry)
    return {
        'schema': 'physicalrsi.egl-public-diagnostics/v1',
        'receipt_sha256': expected_receipt_sha256,
        'trajectory_sha256': trajectory_sha,
        'official_outcome': receipt['outcome'],
        'trajectory': public,
        'boundary': {
            'evaluator_measurements_included': False,
            'depth_source': 'simulator-rendered RGB-D',
            'prior_gt_assisted_development_erased': False,
            'independent_semantic_qualification': None,
        },
    }
