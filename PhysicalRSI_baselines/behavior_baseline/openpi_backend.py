"""Adapter for the official OpenPI BEHAVIOR action backend."""
from copy import deepcopy
import dataclasses
import os
from pathlib import Path
import subprocess

import numpy as np

from PhysicalRSI_core.infra.storage import file_digest, relative_path
from .observations import CAMERAS, PROPRIO

OPENPI_REVISION = "0cc8e355f7bac0976db1cc3139b1ff0379feea60"
TOKENIZER = dict(uri="gs://big_vision/paligemma_tokenizer.model",
                 sha256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6")



def supported_tasks(specification):
    """Declared task coverage, not evidence of task competence."""
    kind = specification.get("kind")
    if kind == "champion-pi05-b1k/v1":
        from .champion_backend import supported_tasks as champion_tasks
        return champion_tasks(specification)
    if kind == "openpi-b1k/v1":
        task = specification.get("task")
        if not isinstance(task, str) or not task or "task_prompts" in specification:
            raise ValueError("Declare exactly one legacy task")
        return frozenset([task])
    if kind != "openpi-b1k-multitask/v1" or "task" in specification:
        raise ValueError("Unsupported or ambiguous OpenPI task binding")
    prompts = specification.get("task_prompts")
    if (not isinstance(prompts, dict) or not prompts
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
                   for k, v in prompts.items())):
        raise ValueError("Pin a nonempty prompt for every declared task")
    asset = specification.get("asset_id")
    if not isinstance(asset, str) or not asset:
        raise ValueError("Declare the shared normalization asset")
    relative_path(asset)
    if "assets/" + asset + "/norm_stats.json" not in specification.get("checkpoint_files", {}):
        raise ValueError("Shared checkpoint lacks declared normalization assets")
    return frozenset(prompts)


def select_task(specification, task=None):
    tasks = supported_tasks(specification)
    task = specification.get("task") if task is None else task
    if task not in tasks:
        raise ValueError("Explicit served task must belong to frozen checkpoint coverage")
    return task


def verify_tokenizer(specification):
    """Require the pinned public tokenizer in OpenPI's actual download cache."""
    if specification.get("tokenizer") != TOKENIZER:
        raise ValueError("Pin the inspected official PaliGemma tokenizer")
    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
    path = cache / "big_vision/paligemma_tokenizer.model"
    if not path.is_file() or file_digest(path) != TOKENIZER["sha256"]:
        raise ValueError("Official tokenizer is missing or changed in OPENPI_DATA_HOME")
    return path


def baseline_observation(public):
    """Restore only the indices consumed by the official baseline transforms.

    Unused simulator fields remain zero and are not available to the program.
    Camera resize, normalization, gripper summation, action decoding and action
    chunking remain the original baseline implementation's responsibilities.
    """
    state = public["proprio"]
    batch = state["base_qvel"].shape[0]
    vector = np.zeros((batch, 256), dtype=np.float32)
    for name, indices in PROPRIO.items():
        values = np.asarray(state[name])
        if values.shape != (batch, len(indices)) or not np.isfinite(values).all():
            raise ValueError("Malformed public proprioception")
        vector[:, indices] = values
    raw = {"robot::proprio": vector}
    for role, prefix in CAMERAS.items():
        raw[prefix + "::rgb"] = public["rgb"][role].copy()
    return raw


class OpenPIBackend:
    def __init__(self, specification, *, task=None):
        self.specification = deepcopy(specification)
        self.task = select_task(specification, task)
        if specification["source_revision"] != OPENPI_REVISION:
            raise ValueError("Use the inspected official BEHAVIOR baseline revision")
        if specification["adapter_sha256"] != file_digest(Path(__file__)):
            raise ValueError("Action adapter implementation differs from the frozen backend")
        verify_tokenizer(specification)
        source = Path(specification["source_root"]).resolve()
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30).strip()
        if revision != OPENPI_REVISION:
            raise ValueError("OpenPI source revision changed")
        subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True, timeout=120)
        checkpoint = Path(specification["checkpoint_root"]).resolve()
        files = specification["checkpoint_files"]
        if not files:
            raise ValueError("Pin checkpoint weights and normalization assets")
        actual_files = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
        if actual_files != set(files):
            raise ValueError("Checkpoint file inventory changed")
        for name, expected in files.items():
            path = checkpoint / relative_path(name)
            if not path.resolve().is_relative_to(checkpoint) or file_digest(path) != expected:
                raise ValueError("Checkpoint content changed")

        # Heavy model dependencies are imported only inside the policy venv.
        import openpi
        from openpi.configs.tasks import TASK_REGISTRY
        from openpi.policies import policy_config
        from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
        from openpi.training import config
        if not Path(openpi.__file__).resolve().is_relative_to(source):
            raise ValueError("Imported OpenPI differs from the pinned checkout")
        task = self.task
        prompt = (specification["task_prompts"][task] if "task_prompts" in specification
                  else TASK_REGISTRY["b1k"][task])
        cfg = config.get_config(specification["training_config"])
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(
            cfg.data, repo_id=specification["repo_id"], robot_config_name="b1k/R1Pro"))
        if "asset_id" in specification:
            cfg = dataclasses.replace(cfg, data=dataclasses.replace(
                cfg.data, assets=config.AssetsConfig(asset_id=specification["asset_id"])))
        model = policy_config.create_trained_policy(cfg, str(checkpoint), default_prompt=prompt)
        self.wrapper = B1KPolicyWrapper(model, "b1k/R1Pro", prompt,
                                       specification["control_mode"], specification["action_horizon"],
                                       cfg.model.action_horizon)

    def identity(self):
        return dict(deepcopy(self.specification), adapter_sha256=file_digest(Path(__file__)))

    def reset(self, task):
        if task != self.task:
            raise ValueError("Served checkpoint/task binding differs from the requested episode")
        self.wrapper.reset()

    def act(self, observation, context):
        context.check()
        action = self.wrapper.act(baseline_observation(observation))
        context.check()
        return action
