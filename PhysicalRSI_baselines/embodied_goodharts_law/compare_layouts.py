"""Compare private object-placement evidence within one native task.

A seed change alone is not layout novelty. This conservative comparison covers
object root poses, not articulated-joint changes, textures or language variants.
Its private geometry must not enter candidate observations or memory.
"""
import argparse
import json
import math
from pathlib import Path

from PhysicalRSI_core.infra.artifacts import Artifacts
from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest


def read_geometry(path):
    import numpy as np

    layout = json.loads(path.read_text())
    if layout.get("schema") != "physicalrsi.egl-layout/v1" or layout.get("state") != "captured":
        raise ValueError("Expected a captured native layout")
    if digest(layout["identity"]) != layout["layout_sha256"]:
        raise ValueError("Layout identity digest mismatch")
    references = layout["private_reset"]["object_root_poses"]
    if not references:
        raise ValueError("Layout contains no object-placement evidence")
    store = Artifacts(path.parent / "artifacts")
    geometry = {name: store.read_array(ref, max_bytes=1024) for name, ref in references.items()}
    for pose in geometry.values():
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1):
            raise ValueError("Expected finite position and unit WXYZ quaternion")
    if digest({name: pose.tolist() for name, pose in geometry.items()}) != layout["identity"]["geometry_sha256"]:
        raise ValueError("Object geometry differs from layout identity")
    return layout, geometry


def compare(left, right, *, translation_m=0.001, rotation_rad=math.pi / 360):
    import numpy as np

    if any(not math.isfinite(value) or value <= 0 for value in (translation_m, rotation_rad)):
        raise ValueError("Use finite positive novelty thresholds")
    a, ga = read_geometry(left)
    b, gb = read_geometry(right)
    for key in ("benchmark", "suite", "task", "bddl_sha256"):
        if a["identity"][key] != b["identity"][key]:
            raise ValueError("Compare layouts within the same pinned task")
    if set(ga) != set(gb):
        raise ValueError("Object population changed; declare a separate layout condition")
    changes = {}
    for name in ga:
        p, q = ga[name], gb[name]
        distance = float(np.linalg.norm(p[:3] - q[:3]))
        cosine = abs(float(np.dot(p[3:], q[3:]) / (np.linalg.norm(p[3:]) * np.linalg.norm(q[3:]))))
        angle = 2 * math.acos(min(1, max(0, cosine)))
        changes[name] = {"translation_m": distance, "rotation_rad": angle,
                         "changed": distance > translation_m or angle > rotation_rad}
    return {"schema": "physicalrsi.egl-layout-comparison/v1", "scope": "private-layout-placement-preflight",
            "qualification": None, "benchmark_trial": False,
            "left_layout_sha256": a["layout_sha256"], "right_layout_sha256": b["layout_sha256"],
            "left_file_sha256": file_digest(left), "right_file_sha256": file_digest(right),
            "thresholds": {"translation_m": translation_m, "rotation_rad": rotation_rad},
            "placement_changed": any(row["changed"] for row in changes.values()), "objects": changes,
            "comparator_sha256": file_digest(Path(__file__)),
            "limitation": "Pairwise object-root placement comparison only; not novelty against all historical layouts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left.resolve(), args.right.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "comparison.json", result)
    print(args.output / "comparison.json")


if __name__ == "__main__":
    main()
