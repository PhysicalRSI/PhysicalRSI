"""Reference policy time and Tianji servo time are different units.

GPT-as-Policy/hybrid_rollout/robodojo/skill/context/eef_control.md declares
25 Hz native control observations. Tianji's existing servo runs at 50 Hz.
"""

SERVO_DT = 0.02
POLICY_DT = 0.04
SERVO_TICKS_PER_POLICY_STEP = 2


def direct_servo_ticks(steps):
    if type(steps) is not int or not 1 <= steps <= 5:
        raise ValueError("Direct control requires 1..5 policy steps at 40 ms each")
    return steps * SERVO_TICKS_PER_POLICY_STEP
