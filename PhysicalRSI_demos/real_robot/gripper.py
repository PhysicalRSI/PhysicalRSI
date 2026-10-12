"""Semantic gripper request contract; profiles and motor control belong to the host."""

MODES = ("wrap", "tripod")

def command(arm, action, mode=None):
    if arm not in ("left", "right") or action not in ("open", "close", "hold", "release"):
        raise ValueError("Gripper requires left/right and open/close/hold/release")
    if mode is not None and mode not in MODES:
        raise ValueError("Grasp mode must be wrap or tripod")
    if action == "hold" and mode is not None:
        raise ValueError("Hold preserves the current grasp; do not select a new mode")
    return dict(arm=arm, action=action, mode=mode)
