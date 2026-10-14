"""Frozen 2025 champion policy, transferred to the public 2026 input boundary."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

from PhysicalRSI_core.infra.storage import file_digest, relative_path
from .champion_observation import (
    SOURCE_REVISION, OPENPI_REVISION, champion_observation,
)

KIND = "champion-pi05-b1k/v1"
UTILITY_SHA256 = "df7e556a6b592f181744c840a459f500f72e79d1485bf3f5d43ae94aa7ad136c"


# Published checkpoint_2 assignment at SOURCE_REVISION. Each candidate binds one task.
CHECKPOINT_2_TASKS = {
    'turning_on_radio': 0,
    'picking_up_trash': 1,
    'picking_up_toys': 7,
    'rearranging_kitchen_furniture': 8,
    'putting_up_Christmas_decorations_inside': 9,
    'preparing_lunch_box': 12,
    'moving_boxes_to_storage': 16,
    'bringing_water': 17,
    'tidying_bedroom': 18,
    'sorting_vegetables': 20,
    'collecting_childrens_toys': 21,
    'putting_shoes_on_rack': 22,
    'assembling_gift_baskets': 26,
    'setting_the_fire': 30,
    'slicing_vegetables': 43,
    'cook_hot_dogs': 45,
}


def supported_tasks(specification):
    task = specification.get("task")
    if (specification.get("kind") != KIND
            or not isinstance(task, str) or task not in CHECKPOINT_2_TASKS
            or type(specification.get("task_id")) is not int
            or specification["task_id"] != CHECKPOINT_2_TASKS[task]
            or specification.get("checkpoint_name") != "checkpoint_2"):
        raise ValueError("Bind one published checkpoint_2 task and its exact legacy task ID")
    return frozenset([task])


def verify(specification):
    """Verify policy sources and weights before importing upstream Python."""
    supported_tasks(specification)
    if specification.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Champion source binding differs")
    if specification.get("adapter_sha256") != file_digest(Path(__file__)):
        raise ValueError("Champion adapter changed")
    if specification.get("projection_sha256") != file_digest(Path(__file__).with_name("champion_observation.py")):
        raise ValueError("Champion observation projection changed")
    source = Path(specification["source_root"]).resolve()
    for root, revision in ((source, SOURCE_REVISION), (source / "openpi", OPENPI_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30).strip()
        if actual != revision:
            raise ValueError("Champion source revision changed")
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "--ignore-submodules", "HEAD", "--"],
                       check=True, timeout=120)
    utility = Path(specification["legacy_utility"]).resolve()
    if file_digest(utility) != UTILITY_SHA256:
        raise ValueError("Pinned legacy observation utility changed")
    checkpoint = Path(specification["checkpoint_root"]).resolve()
    files = specification["checkpoint_files"]
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
    if not files or actual != set(files):
        raise ValueError("Champion checkpoint inventory changed")
    for name, sha in files.items():
        path = checkpoint / relative_path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(checkpoint) or file_digest(path) != sha:
            raise ValueError("Champion checkpoint bytes changed")
    transport = Path(specification["transport_source"]).resolve()
    if file_digest(transport) != specification["transport_sha256"]:
        raise ValueError("Frozen native websocket transport changed")
    return source, checkpoint, utility


def load_policy_sources(source, utility):
    """Load pure upstream observation utilities without importing a simulator."""
    for name in ("openpi", "b1k"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("Run champion in a fresh process with its pinned sources")
    sys.path[:0] = [str(source / "src"), str(source / "openpi/src"),
                    str(source / "openpi/packages/openpi-client/src")]
    name = "omnigibson.learning.utils.eval_utils"
    if name in sys.modules:
        module = sys.modules[name]
        if file_digest(Path(module.__file__)) != UTILITY_SHA256:
            raise ValueError("Legacy observation module differs from the pinned source")
        return
    if "omnigibson" in sys.modules:
        raise ValueError("Policy process must not host a simulator")
    for parent in ("omnigibson", "omnigibson.learning", "omnigibson.learning.utils"):
        module = types.ModuleType(parent)
        module.__path__ = []
        sys.modules[parent] = module
    spec = importlib.util.spec_from_file_location(name, utility)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


class ChampionBackend:
    def __init__(self, specification, *, task=None):
        self.specification = deepcopy(specification)
        self.task = specification["task"] if task is None else task
        if self.task not in supported_tasks(specification):
            raise ValueError("Champion task binding differs")
        source, checkpoint, utility = verify(specification)
        load_policy_sources(source, utility)
        from . import champion_policy_config as policy_config
        from b1k.training.config import get_config
        from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
        model = policy_config.create_trained_policy(
            get_config("pi_behavior_b1k_fast"), checkpoint, sample_kwargs={"num_steps": 20})
        # Preserve the published reference's correction rules and stage voting.
        self.wrapper = B1KPolicyWrapper(model, task_id=specification["task_id"])
        self._episode_execution_mode = None

    def identity(self):
        return dict(deepcopy(self.specification), adapter_sha256=file_digest(Path(__file__)),
                    projection_sha256=file_digest(Path(__file__).with_name("champion_observation.py")))

    def reset(self, task):
        if task != self.task:
            raise ValueError("Champion task binding differs")
        self.wrapper.reset()
        self.wrapper.config.actions_to_execute = 26
        self._episode_execution_mode = None

    def act(self, observation, context):
        return self.act_with_execution(observation, context, execution_mode="published")

    def act_with_execution(self, observation, context, *, execution_mode):
        """Select a predefined timing mode once per episode, using the same weights."""
        context.check()
        if type(execution_mode) is not str or execution_mode not in ("published", "uncompressed"):
            raise ValueError("Unknown champion execution mode")
        if self._episode_execution_mode is None:
            # Both modes predict every20 native steps. The alternative uses
            #20 consecutive targets and retains the following4 for inpainting.
            self.wrapper.config.actions_to_execute = 26 if execution_mode == "published" else 20
            self._episode_execution_mode = execution_mode
        elif execution_mode != self._episode_execution_mode:
            raise ValueError("Champion execution mode cannot change within an episode")
        action = self.wrapper.act(champion_observation(observation))
        context.check()
        return action


def transport_class(specification):
    path = Path(specification["transport_source"])
    if file_digest(path) != specification["transport_sha256"]:
        raise ValueError("Frozen native websocket transport changed")
    spec = importlib.util.spec_from_file_location("champion_native_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebsocketPolicyServer
