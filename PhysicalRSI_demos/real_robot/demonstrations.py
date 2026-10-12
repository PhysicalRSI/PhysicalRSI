"""L3 handoff and measured teleoperation demonstrations, never trajectory replay."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import time

from .contracts import Frame, HumanDecisionPending, Observation
from .evidence import write_new

CAMERA_NAMES = dict(front="cam_high", left="cam_left_wrist", right="cam_right_wrist")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass
class Demonstration:
    observations: list[Observation]
    trace: dict
    manifest: dict


def import_hdf5(path, output, *, fk, keyframe_stride=15, not_before_ns=None):
    """Import the existing collector's 26D measured state + commanded action.

    Full HDF5 and video evidence is retained. Keyframes are model attachments,
    not a replacement for the complete time-aligned state/action trajectory.
    """
    import h5py
    import numpy as np
    from PIL import Image
    source, output = Path(path).resolve(), Path(output).resolve()
    if source.is_symlink() or not source.is_file() or keyframe_stride < 1:
        raise ValueError("A real collector HDF5 file and positive keyframe stride are required")
    output.mkdir(parents=True, exist_ok=False)
    retained = output / "teleoperation.hdf5"
    shutil.copyfile(source, retained)
    with h5py.File(retained, "r") as data:
        required = dict(hand_stack="manus_revo2", qpos_layout="interleaved_LhRh",
                        action_source="armcmd_handcmd", qpos_arm_source="measured", qpos_hand_source="measured")
        for key, expected in required.items():
            actual = data.attrs.get(key)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual != expected:
                raise ValueError(f"Demonstration {key} must be {expected}")
        if int(data.attrs.get("measured_fallback_frames", 0)) != 0:
            raise ValueError("Hand command fallback is not measured demonstration evidence")
        measured = np.asarray(data["observations/qpos"], dtype=float)
        actions = np.asarray(data["action"], dtype=float)
        stamps = np.asarray(data["timestamp_ros_nsec"], dtype=np.int64)
        n = len(stamps)
        if (n < 2 or measured.shape != (n, 26) or actions.shape != (n, 26)
                or not np.isfinite(measured).all() or not np.isfinite(actions).all()
                or np.any(np.diff(stamps) <= 0)):
            raise ValueError("Demonstration requires finite aligned 26D measurements/actions and monotonic timestamps")
        if not_before_ns is not None and int(stamps[0]) < not_before_ns:
            raise ValueError("Demonstration predates the current L3 handoff")
        indices = sorted(set(range(0, n, keyframe_stride)) | {n-1})
        frames = {i: [] for i in indices}
        files = {"teleoperation.hdf5": file_hash(retained)}
        for camera, recorded in CAMERA_NAMES.items():
            dataset = data.get("observations/images/" + recorded)
            if dataset is not None:
                if len(dataset) != n:
                    raise ValueError("HDF5 image/state frame counts differ")
                images = ((i, np.asarray(dataset[i])) for i in indices)
            else:
                import av
                video = source.parent / "video" / recorded / (source.stem + ".mp4")
                target = output / (recorded + ".mp4")
                shutil.copyfile(video, target)
                files[target.name] = file_hash(target)
                # Decode all frames to verify exact alignment, retaining selected
                # RGB frames for model inspection. AV1 is the collector format.
                def decode(video_path=target):
                    count = 0
                    with av.open(str(video_path)) as container:
                        for i, frame in enumerate(container.decode(video=0)):
                            count += 1
                            if i in frames:
                                yield i, frame.to_rgb().to_ndarray()
                    if count != n:
                        raise ValueError("Video/state frame counts differ")
                images = decode()
            for i, rgb in images:
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.any():
                    raise ValueError("Demonstration keyframe is missing or not RGB uint8")
                image = output / "frames" / f"{i:06d}-{camera}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(image)
                frames[i].append(Frame(camera, str(image), int(stamps[i]), file_hash(image)))
        command_matches = data.attrs.get("arm_cmd_matched_frames")
        command_semantics = ("commands" if command_matches is not None and int(command_matches) == n
                             else "arm commands may include measured-position fallback; do not infer unobserved intent")
    observations = [Observation(f"l3-{files['teleoperation.hdf5'][:16]}-{i}", int(stamps[i]), tuple(frames[i]), tuple(measured[i].tolist()),
                                fk(measured[i].tolist())) for i in indices]
    for obs in observations:
        obs.validate()
    trace = dict(timestamps_ns=stamps.tolist(), measured_qpos26=measured.tolist(),
                 commanded_action26=actions.tolist(), command_semantics=command_semantics)
    write_new(output / "trajectory.json", trace)
    files["trajectory.json"] = file_hash(output / "trajectory.json")
    manifest = dict(schema="tianji.l3-demonstration/v1", source=str(source), files=files, frames=n,
                    first_ns=int(stamps[0]), last_ns=int(stamps[-1]), keyframes=indices,
                    timestamp_semantics="collector ROS alignment anchors; camera acquisition timestamps were not recorded",
                    calibration_present=False, replay_allowed=False)
    write_new(output / "manifest.json", manifest)
    return Demonstration(observations, trace, manifest)


class TerminalTeleoperation:
    def __init__(self, *, fk, read=input, write=print, dialogue=None):
        self.fk, self.read, self.write = fk, read, write
        self.dialogue = dialogue

    def run(self, context):
        context.action_revision += 1  # Invalidate pre-takeover completion judgments.
        # release_for_human must confirm stop, close SDKs, release shared locks
        # and terminate the hardware worker. No process-name/lease bypass.
        context.robot.release_for_human()
        start = time.time_ns()
        context.episode.event("human_handoff", level="L3", started_ns=start)
        self.write("\n[L3 人工遥操接管] agent 已交还硬件所有权。")
        self.write("请用现有 PICO/MANUS 遥操并采集本次接管；沿用共享锁。")
        self.write("接续当前场景应使用臂/手遥操的 --no-home 选项；不要运行会自动回 home 的总启动脚本。")
        self.write("完成后停止自己的遥操流程，在这里输入本次采集的 episode_N.hdf5 路径。")
        index = 0
        while True:
            try:
                if self.dialogue:
                    pending = self.dialogue.request("T", "请继续遥操并采集；完成这一段后输入 HDF5 路径。")
                    while not pending.done():
                        context._resolve_monitor(wait=False)
                        time.sleep(.05)
                    path = pending.result().strip()
                else:
                    path = self.read("遥操数据> ").strip()
            except (EOFError, KeyboardInterrupt) as exc:
                raise HumanDecisionPending("Manual teleoperation is pending") from exc
            if not path:
                self.write("任务仍未完成，继续等待本次遥操记录。")
                continue
            index += 1
            try:
                demo = import_hdf5(path, context.episode.public / f"l3-demonstration-{index:04d}",
                                   fk=self.fk, not_before_ns=start)
            except (ValueError, OSError, KeyError) as exc:
                self.write("本段示教未能导入，请检查路径/采集格式：" + str(exc))
                continue
            context.action_revision += 1
            for obs in demo.observations:
                context.episode.observation(obs)
            context.episode.event("human_demonstration", level="L3", trajectory=demo.trace,
                                  manifest=demo.manifest)
            # Record any delayed prior judgment without accepting its old state.
            context._resolve_monitor(wait=True)
            context.checkpoint(force=True, observation=demo.observations[-1])
            context._resolve_monitor(wait=True)
            start = demo.manifest["last_ns"]
            self.write("监控尚未确认完成；L3 保持人工接管，继续下一段，不结束任务。")
