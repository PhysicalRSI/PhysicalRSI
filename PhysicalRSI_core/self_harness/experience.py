"""Owned development experience shared by skill compilation and model training."""
from copy import deepcopy
from pathlib import Path

from ..infra.owned_files import OwnedFiles, physical
from .rollout import development_evaluations, read_development_evaluations


class DevelopmentExperience:
    def __init__(self, store):
        if not isinstance(store, OwnedFiles):
            raise ValueError("Development experience requires an owned artifact store")
        self.store = store

    def capture(self, parent, feedback, source_root):
        source_root = Path(source_root).resolve()
        if self.store.root.is_relative_to(source_root) or source_root.is_relative_to(self.store.root):
            raise ValueError("Keep experience storage outside source experiments")
        values = development_evaluations(parent, feedback, source_root)
        files = {"evidence/" + name: dict(path=str(physical(source_root, name)), sha256=sha)
                 for name, sha in feedback["evidence"].items()}
        for entries in parent["components"].values():
            for name, sha in entries.items():
                files["candidate/" + name] = dict(path=str(physical(Path(parent["root"]), name)), sha256=sha)
        metadata = dict(schema="physicalrsi.development-experience/v1", parent=deepcopy(parent),
            feedback={k: deepcopy(v) for k, v in feedback.items() if k != "evaluations"},
            trials={value["trial"]["id"]: value["trial_sha256"] for value in values},
            source_root=str(source_root), qualification=None)
        revision = self.store.capture(metadata, files)
        self.read(revision)
        return revision

    def read(self, revision):
        return self.read_with_record(revision)["evaluations"]

    def read_with_record(self, revision):
        """Return metadata and evaluations from one fresh owned-experience audit."""
        record = self.store.read(revision)
        metadata = record["metadata"]
        # Earlier compiled skills used the same structure under a narrower name.
        if metadata.get("schema") not in {"physicalrsi.development-experience/v1", "physicalrsi.skill-experience/v1"}:
            raise ValueError("Unknown development experience")
        root = self.store.folder(revision) / "files"
        parent = dict(metadata["parent"], root=str(root / "candidate"))
        output = root / "evidence"
        values = read_development_evaluations(parent, metadata["feedback"], output)
        if {v["trial"]["id"]: v["trial_sha256"] for v in values} != metadata["trials"]:
            raise ValueError("Owned experience trial identities changed")
        return dict(record=record, evaluations=values)
