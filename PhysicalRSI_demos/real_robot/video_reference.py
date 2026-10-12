"""L3 historical RGB reference; never imports HDF5 or obtains a robot trajectory."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .evidence import write_new

CAMERAS = {"front": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}


def video_inventory(directory):
    root = Path(directory).resolve(strict=True)
    if (root / "aloha_mobile_dummy").is_dir():
        root /= "aloha_mobile_dummy"
    video = root / "video"
    names = set(p.name for p in (video / "cam_high").glob("episode_*.mp4"))
    names = sorted(names, key=lambda n: int(Path(n).stem.split("_")[-1]))
    return [{cam: str(video / folder / name) for cam, folder in CAMERAS.items()}
            for name in names if all((video / folder / name).is_file() for folder in CAMERAS.values())]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe_reference(videos):
    """Read decoded frame PTS once, then reuse during targeted queries."""
    if set(videos) != set(CAMERAS):
        raise ValueError("Three reference views required")
    sources, timelines = {}, {}
    for camera, name in videos.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Reference video must be a regular file")
        meta = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_entries", "format=duration:stream=codec_name:frame=best_effort_timestamp_time",
            "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=180).stdout)
        duration = float(meta["format"]["duration"])
        if not 0 < duration < 3600:
            raise ValueError("Invalid demonstration video duration")
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in meta["frames"]]
        if (not timestamps or not all(math.isfinite(t) for t in timestamps)
                or any(a > b for a, b in zip(timestamps, timestamps[1:]))):
            raise ValueError("Reference requires ordered, finite frame presentation timestamps")
        timelines[camera] = timestamps
        sources[camera] = dict(path=str(path.resolve()), sha256=sha256(path), duration_s=duration,
                               codec=meta["streams"][0]["codec_name"], first_frame_pts_s=timestamps[0])
    return sources, timelines


def extract_reference(videos, output, *, samples, start_s=0., end_s=None, probed=None):
    """Model-selected interval/density, retaining actual decoded frame PTS."""
    if type(samples) is not int or not 1 <= samples <= 16:
        raise ValueError("Each query accepts 1..16 temporal samples")
    sources, timelines = probed if probed is not None else probe_reference(videos)
    if set(videos) != set(CAMERAS) or any(str(Path(videos[c]).resolve()) != s["path"] for c,s in sources.items()):
        raise ValueError("Probe does not belong to these videos")
    # Container duration can extend beyond the last decodable frame. Select
    # actual frame ordinals and retain their PTS, including variable-rate video.
    end = min(ts[-1] - ts[0] for ts in timelines.values())
    end_s = end if end_s is None else end_s
    if (type(start_s) not in (int, float) or type(end_s) not in (int, float)
            or not math.isfinite(start_s) or not math.isfinite(end_s)
            or not 0 <= start_s <= end_s <= end):
        raise ValueError(f"Select an interval within actual common frame times 0..{end}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for camera, source in sources.items():
        timestamps = timelines[camera]
        requested = [(start_s + end_s) / 2] if samples == 1 else [
            start_s + (end_s - start_s) * index / (samples - 1) for index in range(samples)]
        selected = [min(range(len(timestamps)), key=lambda n: abs(timestamps[n] - timestamps[0] - t))
                    for t in requested]
        unique = sorted(set(selected))
        expression = "+".join(f"eq(n\\,{n})" for n in unique)
        decoder = ["-c:v", "libdav1d"] if source["codec"] == "av1" else []
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *decoder, "-i", source["path"],
            "-vf", f"select={expression}", "-vsync", "0", "-frames:v", str(len(unique)),
            "-start_number", "0", str(output / f"{camera}-%03d.png")],
            check=True, capture_output=True, timeout=180)
        for sample, ordinal in enumerate(selected):
            target = output / f"{camera}-{unique.index(ordinal):03d}.png"
            seconds = timestamps[ordinal] - timestamps[0]
            frames.append(dict(camera=camera, path=str(target.resolve()), sha256=sha256(target),
                               captured_ns=max(1, round(seconds * 1e9)), video_time_s=seconds,
                               source_frame_index=ordinal, source_pts_s=timestamps[ordinal],
                               sample_index=sample, requested_video_time_s=requested[sample],
                               source_video_sha256=source["sha256"]))
    frames.sort(key=lambda frame: (frame["sample_index"], frame["camera"]))
    if any(sha256(v["path"]) != v["sha256"] for v in sources.values()):
        raise ValueError("Reference changed during extraction")
    reference = dict(schema="tianji.historical-rgb-reference/v1", sources=sources, frames=frames,
                     query=dict(start_s=start_s, end_s=end_s, samples=samples),
                     time_semantics="video-relative; not a current robot observation",
                     permitted_use="derive reusable visual relations, stages and recovery rules",
                     has_motor_or_eef_trajectory=False, replay_allowed=False, current_task_success=False)
    write_new(output / "manifest.json", reference)
    return reference


# Kept as the public import used by the runner; querying lives separately from decoding.
from .video_learning import VideoReferenceL3
