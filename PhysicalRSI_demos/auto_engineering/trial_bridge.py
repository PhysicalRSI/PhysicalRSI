"""Bounded, file-based IPC for the native worker's reset/action boundary."""
from pathlib import Path
import time

from PhysicalRSI_core.infra.storage import atomic_json, read_json


def receive_plan(directory, observation, *, timeout):
    directory = Path(directory)
    atomic_json(directory / "ready.json", dict(ready=True, observation=observation))
    deadline = time.monotonic() + timeout
    while not (directory / "action.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("System 1 did not dispatch the native plan")
        time.sleep(.05)
    request = read_json(directory / "action.json")
    if set(request) != {"memory", "plan"}:
        raise ValueError("Invalid native macro action")
    return request
