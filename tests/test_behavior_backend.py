import numpy as np

from PhysicalRSI_baselines.behavior_baseline.observations import CAMERAS, PROPRIO, project
from PhysicalRSI_baselines.behavior_baseline.openpi_backend import baseline_observation


def test_baseline_adapter_preserves_exact_model_input_coordinates_without_privileged_fields():
    raw = {"robot::proprio": np.arange(256, dtype=np.float32)[None, :]}
    for prefix in CAMERAS.values():
        raw[prefix + "::rgb"] = np.full((1, 8, 8, 4), 123, dtype=np.uint8)
    restored = baseline_observation(project(raw, task="radio"))
    selected = [i for indices in PROPRIO.values() for i in indices]
    np.testing.assert_array_equal(restored["robot::proprio"][:, selected], raw["robot::proprio"][:, selected])
    excluded = sorted(set(range(256)) - set(selected))
    assert not restored["robot::proprio"][:, excluded].any()
    for prefix in CAMERAS.values():
        np.testing.assert_array_equal(restored[prefix + "::rgb"], raw[prefix + "::rgb"][..., :3])
