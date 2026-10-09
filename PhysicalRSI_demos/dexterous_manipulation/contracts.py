"""Shared numeric validation for the bounded simulation action interface."""
import numpy as np

from .common import Rejected


def vector(value, n, label):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise Rejected(f"{label} must contain {n} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise Rejected(f"{label} must contain numbers, not booleans")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise Rejected(f"{label} contains a non-finite number")
    return result

def pose(value):
    p = vector(value.get("position"), 3, "position")
    q = vector(value.get("quaternion_wxyz"), 4, "quaternion_wxyz")
    if abs(np.linalg.norm(q) - 1) > 1e-4:
        raise Rejected("quaternion_wxyz must be unit length")
    return p, q

def object_schema(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)
