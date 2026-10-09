"""Verify recorded observation and image bytes before interpreting outcomes."""
import json
from pathlib import Path

from .common import digest, file_hash


def verify_media(trace):
    count = 0
    for step in trace:
        observation = step["observation"]
        if json.loads(Path(observation["observation_path"]).read_text()) != observation:
            raise ValueError("Observation evidence changed")
        for image in observation["images"]:
            if file_hash(image["path"]) != image["sha256"]:
                raise ValueError("RGB evidence changed")
            count += 1
    return dict(verified_images=count, trace_sha256=digest(trace))
