"""Exercise ASPIRE reset method bodies without installing simulator dependencies.

This is a targeted software regression check, not a simulation or task score.
Only run against a trusted ASPIRE source checkout: the selected method bodies
are compiled and executed. The explicit path must match the reviewed revision.
"""
import argparse
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np

REVISION = "f4c8939aab0af9b97690c561bd80e282940f7886"
HANDLE = "aspire/sim/cap/integrations/libero/__init__.py"
SIMULATOR = "aspire/sim/cap/envs/simulators/libero.py"


def reset_method(source, class_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"np": np, "Any": Any}
    exec(compile(ast.fix_missing_locations(module), "<trusted-ASPIRE-reset>", "exec"), namespace)
    return namespace["reset"]


class ResetFixture:
    def __init__(self):
        self.position = -1
        self.sim = SimpleNamespace(data=SimpleNamespace(
            qpos=np.zeros(7), xquat=np.ones((1, 4)), xpos=np.zeros((1, 3))))

    def seed(self, seed):
        pass

    def observation(self):
        return {"robot0_joint_pos": np.full(7, self.position), "position": self.position}

    def reset(self):
        self.position = -1
        return self.observation()

    def set_init_state(self, state):
        self.position = state
        return self.observation()


def check(root):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != REVISION:
        raise ValueError("Use the reviewed ASPIRE revision")
    failures = []
    for original in (True, False):
        def read(path):
            if original:
                return subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=root, text=True)
            return (root / path).read_text()
        handle_reset = reset_method(read(HANDLE), "LiberoHandle")
        simulator_reset = reset_method(read(SIMULATOR), "FrankaLiberoEnv")
        broken = []
        for seed, target in [(1, 10), (2, 20)]:
            env = ResetFixture()
            handle = SimpleNamespace(env=env, init_states=[10, 20], task_language="fixture")
            handle.reset = lambda seed=None: handle_reset(handle, seed)
            observation, _ = handle.reset(seed)
            handle_consistent = observation["position"] == env.position == 10
            simulator = SimpleNamespace(handle=handle, gripper_link_idx=0, viser_debug=False,
                                        _step_once=lambda: None, get_observation=env.observation)
            observation, _ = simulator_reset(simulator, seed=seed)
            consistent = (observation["position"] == env.position == target
                          and np.all(simulator.home_joint_position == target))
            broken.append(not (handle_consistent and consistent))
        if original:
            if not all(broken):
                raise AssertionError("Original source no longer reproduces the regression")
        elif any(broken):
            failures.append("Patched reset failed to preserve selected state and observations")
    if failures:
        raise AssertionError("; ".join(failures))
    print("Original reset regression reproduced; patched methods preserve both selected layouts.")
    print("Scope: software fixture only; no simulator or model executed; qualification: null.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspire-root", type=Path, required=True)
    check(parser.parse_args().aspire_root.resolve())
