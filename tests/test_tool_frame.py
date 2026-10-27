import math
import pytest
from physicalrsi.skills.tool_frame import command_pose_for_tool,tool_frame_transform


def test_observed_tool_offset_corrects_top_down_command_height():
    # A general rigid tool offset: no task name, seed or fixed correction in skill.
    command=[0,0,1,1,0,0,0];tool=[.12,0,1,1,0,0,0]
    desired=[.2,.1,.9,-.5,.5,-.5,-.5]
    result=command_pose_for_tool(desired,command,tool)
    assert result[:3]==pytest.approx([.2,.1,1.02])
    assert result[3:]==pytest.approx(desired[3:])


def test_rotated_tool_frame_and_zero_offset():
    q=[math.sqrt(.5),0,0,math.sqrt(.5)]
    result=command_pose_for_tool([1,2,3,*q],[0,0,0,1,0,0,0],[.2,0,0,*q])
    assert result==pytest.approx([.8,2,3,1,0,0,0])
    pose=[1,2,3,*q]
    assert command_pose_for_tool(pose,pose,pose)==pytest.approx(pose)


def test_skill_observes_different_embodiment_offsets():
    skill=tool_frame_transform()
    for offset in [.03,.12,.2]:
        value=dict(target_tool_pose=[1,0,0,1,0,0,0],observed_command_pose=[0,0,0,1,0,0,0],observed_tool_pose=[offset,0,0,1,0,0,0])
        assert skill.execute(value)['command_pose'][0]==pytest.approx(1-offset)
    with pytest.raises(ValueError):command_pose_for_tool([0]*7,[0]*7,[0]*7)
