import numpy as np
import pytest
from physicalrsi.runtime.sensors import encode_rgb
from physicalrsi.runtime.rgbd import encode_depth_m
from physicalrsi.skills.visible_surface import visible_surface


def packet_fixture():
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[3:6, 4:7] = [100, 0, 0]
    rgb[8:11, 4:7] = [100, 0, 0]
    base = dict(schema='physicalrsi.sensors/v1', sequence=0, timestamp_s=1.,
                embodiment_revision='fixture', calibration_revision='rgbd-fixture',
                cameras={'head_camera': {'captured_at_s': 1., 'rgb_png_b64': encode_rgb(rgb)}},
                actuator_position=[0.])
    return dict(schema='physicalrsi.rgbd/v1', rgb=base, depth_model='fixture-depth',
                depth={'head_camera': {'captured_at_s': 1., 'encoding': 'png-u16-mm',
                    'invalid_value': 0, 'minimum_mm': 100, 'maximum_mm': 3000,
                    'png_b64': encode_depth_m(np.full((12, 16), 1.234))}})


def fixture():
    packet = packet_fixture()
    skill = visible_surface(camera="head_camera", intrinsic_cv=[[10, 0, 0], [0, 10, 0], [0, 0, 1]],
        extrinsic_cv=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        image_size_px=[16, 12], calibration_revision="rgbd-fixture",
        depth_model=packet["depth_model"])
    proposal = dict(sequence=packet["rgb"]["sequence"], camera="head_camera",
                    uv=[5.5/16, 4.5/12], bbox_xyxy=[0, 0, 16, 12])
    return skill, dict(observation=packet, proposal=proposal)


def test_connected_patch_excludes_other_object_and_preserves_metric_depth():
    skill, value = fixture()
    result = skill.execute(value)
    assert set(result["region_flat_indices"]) == {y*16+x for y in range(3, 6) for x in range(4, 7)}
    assert result["centroid_world_m"] == pytest.approx([.55*1.234, .45*1.234, 1.234])
    assert result["semantic_identity"] is None


@pytest.mark.parametrize("change", ["stale", "oracle", "camera", "outside"])
def test_rejects_unbound_or_extra_inputs(change):
    skill, value = fixture()
    if change == "stale":
        value["proposal"]["sequence"] += 1
    elif change == "oracle":
        value["observation"]["object_pose"] = [0, 0, 0]
    elif change == "camera":
        value["proposal"]["camera"] = "other"
    else:
        value["proposal"]["bbox_xyxy"] = [0, 0, 2, 2]
    with pytest.raises(ValueError):
        skill.execute(value)


def test_missing_depth_abstains():
    skill, value = fixture()
    value["observation"]["depth"]["head_camera"]["png_b64"] = encode_depth_m(np.zeros((12, 16)))
    assert skill.execute(value)["status"] == "unavailable"
