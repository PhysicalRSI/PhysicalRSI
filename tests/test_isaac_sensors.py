"""Sensor startup contract tests; no native rendering qualification."""
from types import SimpleNamespace

import pytest

from PhysicalRSI_demos.auto_engineering import isaac_scene


def test_sensor_readiness_waits_for_all_outputs_without_physics_steps(monkeypatch):
    world = SimpleNamespace(renders=0, current_time_step_index=92, current_time=1.5)
    def render():
        world.renders += 1
    monkeypatch.setattr(isaac_scene, "capture_sensor_frame", render)
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
    assert warmup["physics_step"] == 92 and warmup["physics_time"] == 1.5


def test_missing_sensor_data_times_out_before_dispatch():
    camera = SimpleNamespace(get_resolution=lambda: (8, 6), get_rgba=lambda: None,
                             get_depth=lambda: SimpleNamespace(shape=(0,)))
    with pytest.raises(TimeoutError, match="did not become ready"):
        isaac_scene.initial_sensor_frames(
            SimpleNamespace(current_time_step_index=92, current_time=1.5), camera, camera, timeout=0)


@pytest.mark.parametrize('field,value', [('current_time_step_index', 93), ('current_time', 1.6)])
def test_sensor_capture_rejects_physics_advance_even_if_images_become_ready(monkeypatch, field, value):
    world = SimpleNamespace(current_time_step_index=92, current_time=1.5, captured=False)
    def capture():
        setattr(world, field, value)
        world.captured = True
    monkeypatch.setattr(isaac_scene, 'capture_sensor_frame', capture)
    camera = SimpleNamespace(get_resolution=lambda: (8, 6),
        get_rgba=lambda: SimpleNamespace(shape=(6, 8, 4) if world.captured else (0,)),
        get_depth=lambda: SimpleNamespace(shape=(6, 8) if world.captured else (0,)))
    with pytest.raises(RuntimeError, match='advanced settled physics'):
        isaac_scene.initial_sensor_frames(world, camera, camera)
