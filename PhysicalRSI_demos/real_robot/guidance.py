"""Operator language assistance is L3 execution evidence, never a judge answer."""
import json
from pathlib import Path
import uuid

from .evidence import write_new


def submit(directory, text):
    root = Path(directory).resolve(strict=True)
    if (root / "episode").is_dir():
        root /= "episode"
    metadata = json.loads((root / "episode.json").read_text())
    if metadata.get("split") != "development" or (root / "result.json").exists():
        raise ValueError("Guidance requires an unfinished development episode")
    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
        raise ValueError("Provide concise public operator guidance")
    path = root / "execution" / "guidance-inbox" / (uuid.uuid4().hex + ".json")
    write_new(path, dict(kind="human_language_guidance", guidance=text.strip()))
    return path


def consume(episode):
    if episode.split != "development":
        return
    seen = {e.get("guidance_id") for e in episode.events if e["kind"] == "human_guidance"}
    for path in sorted((episode.public / "guidance-inbox").glob("*.json")):
        if path.name in seen:
            continue
        if path.is_symlink():
            raise ValueError("Guidance cannot reference evaluator files")
        value = json.loads(path.read_text())
        if (set(value) != {"kind", "guidance"} or value["kind"] != "human_language_guidance"
                or not isinstance(value["guidance"], str) or not 0 < len(value["guidance"]) <= 8000):
            raise ValueError("Invalid public language guidance")
        episode.event("human_guidance", level="L3", guidance_id=path.name,
                      guidance=value["guidance"], subtype="language")
