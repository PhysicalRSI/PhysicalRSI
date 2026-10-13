"""Sensor startup contract tests; no native rendering qualification."""
from types import SimpleNamespace

import pytest

from PhysicalRSI_demos.auto_engineering import isaac_scene


def test_sensor_readiness_waits_for_all_outputs_without_physics_steps(monkeypatch):
    world = SimpleNamespace(renders=0)
    def render():
        world.renders += 1
    world.render = render
    monkeypatch.setattr(isaac_scene.time, "sleep", lambda _: None)
    overhead = SimpleNamespace(
        get_resolution=lambda: (8, 6),
        get_rgba=lambda: SimpleNamespace(shape=(6, 8, 4) if world.renders >= 1 else (0,)),
        get_depth=lambda: SimpleNamespace(shape=(6, 8) if world.renders >= 2 else (0,)))
    view = SimpleNamespace(get_resolution=lambda: (16, 9),
        get_rgba=lambda: SimpleNamespace(shape=(9, 16, 4) if world.renders >= 3 else (0,)))
    rgb, depth, presentation, warmup = isaac_scene.initial_sensor_frames(world, overhead, view)
    assert (rgb.shape, depth.shape, presentation.shape) == ((6, 8, 4), (6, 8), (9, 16, 4))
    assert warmup["renders"] == world.renders == 3


def test_missing_sensor_data_times_out_before_dispatch():
    camera = SimpleNamespace(get_resolution=lambda: (8, 6), get_rgba=lambda: None,
                             get_depth=lambda: SimpleNamespace(shape=(0,)))
    with pytest.raises(TimeoutError, match="did not become ready"):
        isaac_scene.initial_sensor_frames(object(), camera, camera, timeout=0)
