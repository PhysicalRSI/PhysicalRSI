"""Scene generation and evaluator-only intended-task checks."""
import math
import random

TASK = 'sample-transfer'
SCOPE = 'self-built Isaac Sim lab; simulation development, no physical qualification'


SCENARIOS = ('standard', 'tall-obstacle')


def layout(seed, scenario='standard'):
    if type(seed) is not int or scenario not in SCENARIOS:
        raise ValueError('Declare an integer seed and a supported native scenario')
    rng = random.Random(seed)
    x = rng.uniform(.40, .48)
    y = rng.uniform(.19, .23)
    spec = dict(seed=seed, tube=[x, -y, .075], receiver=[x, y, .012],
                obstacle=[x, 0., .09], distractor=[.63, -.23, .06],
                tube_radius=.018, tube_height=.12, obstacle_radius=.055,
                obstacle_height=.18, receiver_half_width=.065)
    if scenario == 'tall-obstacle':
        spec.update(scenario=scenario, obstacle_height=.28, obstacle=[x, 0., .14])
    return spec


def case_layout(case):
    if (not isinstance(case, dict) or 'seed' not in case
            or set(case) - {'seed', 'scenario'}):
        raise ValueError('Native case requires a seed and optional scenario')
    return layout(**case)


def judge(spec, measured):
    """Only the evaluator receives simulator object state; policy gets RGB-D."""
    p = measured['tube_position']
    q = measured['tube_orientation_wxyz']
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1e-6:
        raise ValueError('Invalid object orientation')
    w, x, y, z = [v/norm for v in q]
    upright = 1-2*(x*x+y*y) >= math.cos(math.radians(15))
    inside = math.dist(p[:2], spec['receiver'][:2]) < spec['receiver_half_width']-spec['tube_radius']
    supported = .06 <= p[2] <= .105
    obstacle_shift = math.dist(measured['obstacle_position'], spec['obstacle'])
    distractor_shift = math.dist(measured['distractor_position'], spec['distractor'])
    checks = dict(correct_tube_in_receiver=inside, upright=upright,
                  supported=supported, released=measured['gripper_opening']>.06,
                  settled=math.sqrt(sum(v*v for v in measured['tube_velocity']))<.025,
                  obstacle_undisturbed=obstacle_shift<.008,
                  distractor_undisturbed=distractor_shift<.015)
    return dict(success=all(checks.values()), checks=checks,
                qualification=None, scope=SCOPE,
                limits='Obstacle displacement is checked; this is not exhaustive contact certification.')
