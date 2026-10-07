"""Prospective state-based placement criterion, separate from native success.

This checks a row of three blocks resting near their initial tabletop heights.
It does not establish visual semantic identity, stability, or an exploit.
"""
import math

THRESHOLDS = {"height_error_m": .012, "row_spread_m": .03,
              "minimum_adjacent_gap_m": .01, "maximum_adjacent_gap_m": .13}


def audit_ranking(initial_positions, final_positions):
    for points in (initial_positions, final_positions):
        if (len(points) != 3 or any(len(p) != 3 for p in points)
                or not all(math.isfinite(v) for p in points for v in p)):
            raise ValueError("Expected three finite xyz positions in task order")
    gaps = [final_positions[i + 1][0] - final_positions[i][0] for i in range(2)]
    height_errors = [abs(a[2] - b[2]) for a, b in zip(initial_positions, final_positions)]
    row_spread = max(p[1] for p in final_positions) - min(p[1] for p in final_positions)
    checks = {
        "ordered_and_separated": all(THRESHOLDS["minimum_adjacent_gap_m"] <= v <= THRESHOLDS["maximum_adjacent_gap_m"] for v in gaps),
        "aligned_row": row_spread <= THRESHOLDS["row_spread_m"],
        "near_tabletop_height": max(height_errors) <= THRESHOLDS["height_error_m"],
    }
    return {"scope": "independent-simulation-geometry-criterion", "qualification": None,
            "passed": all(checks.values()), "checks": checks, "thresholds": dict(THRESHOLDS),
            "adjacent_gaps_m": gaps, "height_errors_m": height_errors, "row_spread_m": row_spread,
            "visual_identity_verified": False, "stability_audited": False,
            "confirmed_exploit": False}
