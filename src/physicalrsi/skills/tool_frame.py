"""Rigid tool-frame calibration from observed poses, independent of embodiment.

Pose convention: world xyz metres followed by a wxyz quaternion. The pure
transform is also loadable by isolated Python simulator workers without Core.
"""
import math


def _pose(value):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError('Expected xyz+wxyz pose')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):
        raise ValueError('Pose entries must be finite numbers')
    length=math.sqrt(sum(x*x for x in value[3:]))
    if length < 1e-12:
        raise ValueError('Quaternion must have nonzero norm')
    return list(value[:3]),[x/length for x in value[3:]]


def _conjugate(q):
    return [q[0],-q[1],-q[2],-q[3]]


def _multiply(a,b):
    w,x,y,z=a;v,i,j,k=b
    return [w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v]


def _rotate(q,v):
    return _multiply(_multiply(q,[0.,*v]),_conjugate(q))[1:]


def command_pose_for_tool(target_tool_pose, observed_command_pose, observed_tool_pose):
    """Solve T_world_command = T_world_tool_target * inv(T_command_tool).

    The two observed poses must be synchronized measurements of a rigid tool.
    Their relative transform is inferred, not fitted to task outcomes.
    """
    target,tq=_pose(target_tool_pose)
    command,cq=_pose(observed_command_pose)
    tool,oq=_pose(observed_tool_pose)
    relative_q=_multiply(_conjugate(cq),oq)
    relative_p=_rotate(_conjugate(cq),[tool[i]-command[i] for i in range(3)])
    desired_q=_multiply(tq,_conjugate(relative_q))
    displacement=_rotate(desired_q,relative_p)
    return [target[i]-displacement[i] for i in range(3)]+desired_q


def tool_frame_transform():
    from pathlib import Path
    from PhysicalRSI_core.infra.storage import digest,file_digest
    from .protocol import CallableSkill,SkillDescriptor
    metadata=dict(frame='world',pose='xyz+wxyz',calibration='synchronized rigid command/tool observations')
    revision=digest(dict(source=file_digest(Path(__file__)),metadata=metadata))
    return CallableSkill(SkillDescriptor('tool_frame_transform',revision,
        dict(name='observed_tool_target/v1',unit='SI',frame='world'),
        dict(name='command_pose/v1',unit='SI',frame='world'),metadata=metadata),
        lambda value,context:dict(command_pose=command_pose_for_tool(
            value['target_tool_pose'],value['observed_command_pose'],value['observed_tool_pose'])))
