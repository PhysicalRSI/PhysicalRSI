"""Read-only, post-episode grasp review; outside the frozen campaign source."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from PhysicalRSI_demos.dexterous_manipulation.common import file_hash
from PhysicalRSI_demos.dexterous_manipulation.task_diagnostics import diagnostic_inputs


def longest(values):
    best = count = 0
    for value in values:
        count = count + 1 if value else 0
        best = max(best, count)
    return best


def review(trajectory, source):
    spec, trace, transitions, inputs, dependency = diagnostic_inputs(trajectory, source)
    assert spec['task'] == 'bimanual_assembly'
    sys.path.insert(0, str(Path(source).resolve() / 'dexjoco'))
    from dexjoco.tasks.mappings import CONFIG_MAPPING
    seed = spec['case']['seed']
    env = CONFIG_MAPPING[spec['task']]().get_environment(
        policy_mode=True, render_mode='rgb_array', randomize=False,
        seed=seed, realtime=False, image_obs=False)
    checkpoints = {x['observation']['native_control_steps']: x['observation']['state'] for x in trace}
    errors, rows = [], []

    def check(step, raw):
        if step in checkpoints:
            error = float(np.max(np.abs(np.asarray(raw['state'][:46]) - checkpoints[step])))
            assert np.isfinite(error) and error <= 1e-7, (step, error)
            errors.append(error)

    try:
        raw, _ = env.reset(seed=seed)
        check(0, raw)
        m, d = env.unwrapped.model, env.unwrapped.data
        tray = m.body('industreal_tray_insert_round_peg_8mm').id
        table = m.geom('table_collision').id
        palm = m.body('allegro_palm_left').id
        tray_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tray
                      and (m.geom_contype[i] or m.geom_conaffinity[i])}
        assert tray_geoms and all(m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX for i in tray_geoms)
        hand_bodies = {palm}
        for i in range(m.nbody):
            if m.body_parentid[i] in hand_bodies:
                hand_bodies.add(i)
        hand_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] in hand_bodies}
        assert hand_geoms

        def sample(step):
            normal = d.geom_xmat[table].reshape(3, 3)[:, 2]
            top = d.geom_xpos[table] + normal * m.geom_size[table, 2]
            clearance = min(float(normal @ (d.geom_xpos[g] - top)
                - np.abs(normal @ d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]) for g in tray_geoms)
            forces = {'table': 0.0, 'left_hand': 0.0}
            for i in range(d.ncon):
                contact = d.contact[i]
                a, b = int(contact.geom1), int(contact.geom2)
                other = b if a in tray_geoms else a if b in tray_geoms else None
                if other is None:
                    continue
                role = 'table' if other == table else 'left_hand' if other in hand_geoms else None
                if role:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, wrench)
                    forces[role] += max(0.0, float(wrench[0]))
            return dict(step=step, tray_lowest_clearance_mm=clearance * 1000,
                        tray_table_normal_force_n=forces['table'],
                        left_hand_tray_normal_force_n=forces['left_hand'])

        initial = sample(0)
        for step, expected in enumerate(transitions, 1):
            assert expected['physics_step'] == step
            raw, _, terminated, truncated, info = env.step(np.asarray(expected['action'], dtype=float))
            check(step, raw)
            assert (bool(info['succeed']), bool(terminated), bool(truncated)) == (
                expected['native_success'], expected['terminated'], expected['truncated'])
            rows.append(sample(step))
    finally:
        env.close()
    assert len(errors) == len(checkpoints)
    # Force distinguishes active supporting contact from a merely listed near contact.
    touching = [r for r in rows if r['left_hand_tray_normal_force_n'] > 1e-5]
    first = touching[0]['step'] if touching else None
    after = [r for r in rows if first is not None and r['step'] >= first]
    thresholds = {}
    for mm in (1, 5, 10, 20):
        flags = [r['tray_lowest_clearance_mm'] >= mm
                 and r['left_hand_tray_normal_force_n'] > 1e-5
                 and r['tray_table_normal_force_n'] <= 1e-5 for r in rows]
        thresholds[str(mm)] = dict(steps=sum(flags), longest_consecutive_steps=longest(flags))
    return dict(schema='assembly.postepisode.left-grasp-review/v1',
        binding=dict(inputs=inputs, native_dependencies_sha256=dependency,
                     review_implementation_sha256=file_hash(__file__)),
        role='Privileged post-episode simulation diagnosis; not an acting-policy sensor or selection metric',
        task=spec['task'], seed=seed, native_success=bool(transitions[-1]['native_success']),
        matched_checkpoints=len(errors), maximum_checkpoint_error=max(errors),
        force_presence_threshold_n=1e-5, initial_clearance_mm=initial['tray_lowest_clearance_mm'],
        first_left_hand_contact_step=first, left_hand_contact_steps=len(touching),
        after_first_left_contact=dict(steps=len(after),
            table_support_steps=sum(r['tray_table_normal_force_n'] > 1e-5 for r in after),
            maximum_clearance_mm=max((r['tray_lowest_clearance_mm'] for r in after), default=None)),
        lift_with_left_contact_and_no_table_support_by_clearance_mm=thresholds,
        final=rows[-1], samples=rows,
        interpretation='Contact alone does not establish a grasp. Height alone does not establish retention. '
          'The tabulated lift/contact intervals test a necessary observable condition only; '
          'they do not certify force closure or change the native task verdict.',
        changes_native_score=False, qualification=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'Preserve previous reviews'
    result = review(args.trajectory, args.source)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'samples', 'binding'}}, indent=2))
