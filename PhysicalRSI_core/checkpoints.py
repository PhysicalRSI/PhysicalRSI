"""Content-addressed inference bundles and verified training provenance.

Backends are caller-supplied trusted implementations. A checkpoint never names
an import, starts training, selects a survivor or supplies physical qualification.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import os
import shutil
import tempfile

from .contracts import Contract, Operation
from .infra.owned_files import OwnedFiles, physical
from .infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path
from .self_harness.artifacts import HarnessSnapshots, verify_harness
from .self_harness.experience import DevelopmentExperience
from .skills import operation_identity


def validate_model(model):
    if (not isinstance(model, dict) or set(model) != {"name", "format", "architecture", "backend", "input", "output", "effects"}
            or not isinstance(model["format"], str) or not model["format"]
            or not isinstance(model["architecture"], dict) or not model["architecture"]
            or not isinstance(model["backend"], dict) or not model["backend"]
            or not isinstance(model["effects"], list) or model["effects"] != sorted(set(model["effects"]))):
        raise ValueError("Declare a complete immutable model interface and backend")
    identifier(model["name"])
    for key in ("input", "output"):
        if asdict(Contract(**model[key])) != model[key]:
            raise ValueError("Declare an exact model operation contract")
    if any(not isinstance(v, str) or not v for v in model["effects"]):
        raise ValueError("Declare model effects explicitly")
    digest(model)


def validate_budget(budget, costs=None):
    if (not isinstance(budget, dict) or set(budget) != {"max_updates", "max_samples", "max_seconds"}
            or any(type(budget[k]) is not int or budget[k] < 1 for k in ("max_updates", "max_samples"))
            or type(budget["max_seconds"]) not in {int, float}
            or not math.isfinite(budget["max_seconds"]) or budget["max_seconds"] <= 0):
        raise ValueError("Declare bounded training updates, samples and duration")
    if costs is not None:
        if (not isinstance(costs, dict) or set(costs) != {"updates", "samples", "seconds"}
                or any(type(costs[k]) is not int or not 1 <= costs[k] <= budget["max_" + k] for k in ("updates", "samples"))
                or type(costs["seconds"]) not in {int, float} or not math.isfinite(costs["seconds"])
                or not 0 <= costs["seconds"] <= budget["max_seconds"]):
            raise ValueError("Training costs exceed the frozen budget or lack measured accounting")


def _operation(model, operation):
    value = operation_identity(operation)
    if any(value[k] != model[k] for k in ("name", "input", "output", "effects")):
        raise ValueError("Checkpoint backend changed the fixed model interface")
    return value


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.checkpoints = OwnedFiles(self.root / "checkpoints")
        self.experiences = DevelopmentExperience(OwnedFiles(self.root / "experience"))

    def identity(self):
        return dict(schema="physicalrsi.checkpoint-store/v1", root=str(self.root),
            implementation=file_digest(Path(__file__)), storage=file_digest(Path(__file__).parent / "infra/owned_files.py"),
            experience=file_digest(Path(__file__).parent / "self_harness/experience.py"),
            operations=file_digest(Path(__file__).with_name("skills.py")))

    def publish(self, *, model, backend, root, entrypoint, files, training=None):
        """Import a reviewed seed or a coordinator-verified training result."""
        validate_model(model)
        if backend.identity() != model["backend"]:
            raise ValueError("Checkpoint backend differs from the model declaration")
        root = Path(root).resolve()
        if not isinstance(files, dict) or not files or entrypoint not in files:
            raise ValueError("Declare the complete inference bundle and entry point")
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
        if actual != set(files):
            raise ValueError("Checkpoint bundle differs from its exact file closure")
        exports = {name:dict(path=str(physical(root, name)), sha256=sha) for name,sha in files.items()}
        for name, sha in files.items():
            if file_digest(physical(root, name)) != sha:
                raise ValueError("Checkpoint changed before backend admission")
        operation = backend.build(root, entrypoint, deepcopy(model))
        metadata = dict(schema="physicalrsi.checkpoint/v1", store=self.identity(), model=deepcopy(model),
                        entrypoint=entrypoint, operation=_operation(model, operation), training=deepcopy(training), qualification=None)
        if backend.identity() != model["backend"]:
            raise ValueError("Checkpoint backend changed during admission")
        self._training(metadata, files)
        revision = self.checkpoints.capture(metadata, exports)
        self.read(revision)
        return revision

    def _training(self, metadata, files):
        training = metadata["training"]
        if training is None:
            return None
        if set(training) != {"request", "handle", "status", "result"}:
            raise ValueError("Incomplete training provenance")
        request, handle, status, result = (training[k] for k in ("request", "handle", "status", "result"))
        if (request["schema"] != "physicalrsi.training-request/v1" or request["store"] != self.identity()
                or request["model"] != metadata["model"] or not request["provider"]
                or request["kind"] not in {"rl", "supervised"}
                or any(row.get("request_sha256") != digest(request) for row in (handle, status, result))
                or not isinstance(handle.get("job_id"), str) or not handle["job_id"]
                or any(row.get("job_id") != handle["job_id"] for row in (status, result))
                or status.get("state") != "completed" or status.get("quiescent") is not True
                or result.get("role") != "training_only" or result["files"] != files
                or result["entrypoint"] != metadata["entrypoint"]):
            raise ValueError("Checkpoint differs from its completed training job")
        validate_budget(request["budget"], result["costs"])
        self.training_source(request)
        return request["parent_checkpoint"]

    def training_source(self, request):
        """Verify on-parent development data before spending training resources."""
        audited = self.experiences.read_with_record(request["experience_revision"])
        values = audited["evaluations"]
        source = audited["record"]["metadata"]["parent"]
        if digest(dict(components=source["components"])) != request["parent_harness_sha256"]:
            raise ValueError("Training source differs from the frozen parent")
        if request["binding_path"] not in source["components"]["weights"]:
            raise ValueError("Training source has no declared model binding")
        path = self.experiences.store.folder(request["experience_revision"]) / "files/candidate"
        binding = read_json(physical(path, request["binding_path"]))
        if binding.get("checkpoint_revision") != request["parent_checkpoint"] or binding.get("store") != self.identity():
            raise ValueError("Training source used another checkpoint")
        expected = dict(kind="model", name=request["model"]["name"], revision=request["parent_checkpoint"])
        for value in values:
            binding = value["receipt"].get("binding")
            if (not binding or expected not in binding["system1"]["dependencies"]
                    or value["trial"]["system2"]["candidate_sha256"] != request["parent_harness_sha256"]):
                raise ValueError("Training requires development trials using the exact parent model")
        return values

    def read(self, revision):
        records, current, expected_model = {}, revision, None
        while current is not None:
            if current in records:
                raise ValueError("Cyclic checkpoint ancestry")
            record = self.checkpoints.read(current)
            value = record["metadata"]
            if (value.get("schema") != "physicalrsi.checkpoint/v1" or value["store"] != self.identity()
                    or value["qualification"] is not None or value["entrypoint"] not in record["files"]):
                raise ValueError("Unknown checkpoint or changed store")
            validate_model(value["model"])
            if expected_model is not None and value["model"] != expected_model:
                raise ValueError("Training changed the fixed model family")
            expected_model = value["model"]
            records[current] = record
            parent = self._training(value, record["files"])
            current = parent
        return deepcopy(records[revision])

    def materialize(self, revision, output):
        record = self.read(revision)
        owned = OwnedFiles(output)
        if owned.root.is_relative_to(self.root) or self.root.is_relative_to(owned.root):
            raise ValueError("Materialize inference bundles outside the checkpoint store")
        root = self.checkpoints.folder(revision) / "files"
        copied = owned.capture(record["metadata"], {name:dict(path=str(root / name),sha256=sha) for name,sha in record["files"].items()})
        if copied != revision:
            raise ValueError("Copied checkpoint identity changed")
        return owned.folder(copied)

    def binding(self, harness, *, binding_path="weights.json", model=None):
        verify_harness(harness)
        root = Path(harness["root"])
        if binding_path not in harness["components"]["weights"]:
            raise ValueError("Declare the checkpoint binding in weights")
        value = read_json(physical(root, binding_path))
        if (set(value) != {"schema", "store", "checkpoint_revision", "package"}
                or value["schema"] != "physicalrsi.checkpoint-binding/v1" or value["store"] != self.identity()):
            raise ValueError("Unknown checkpoint binding")
        package = relative_path(value["package"])
        if str(package) != value["package"] or package.name != value["checkpoint_revision"]:
            raise ValueError("Checkpoint package path differs from its revision")
        record = self.read(value["checkpoint_revision"])
        if model is not None and record["metadata"]["model"] != model:
            raise ValueError("Checkpoint changed the frozen model specification")
        copied = OwnedFiles(root / package.parent)
        if copied.read(value["checkpoint_revision"]) != record:
            raise ValueError("Candidate model differs from its published checkpoint")
        expected = {binding_path:file_digest(root / binding_path),
                    str(package / "record.json"):file_digest(root / package / "record.json"),
                    **{str(package / "files" / name):sha for name,sha in record["files"].items()}}
        if harness["components"]["weights"] != expected:
            raise ValueError("Candidate weights do not cover the exact inference closure")
        return dict(revision=value["checkpoint_revision"], record=record, root=str(root / package / "files"))

    def bind(self, harness, *, backend, binding_path="weights.json", model=None):
        value = self.binding(harness, binding_path=binding_path, model=model)
        metadata = value["record"]["metadata"]
        if backend.identity() != metadata["model"]["backend"]:
            raise ValueError("Runtime model backend changed")
        operation = backend.build(Path(value["root"]), metadata["entrypoint"], deepcopy(metadata["model"]))
        if _operation(metadata["model"], operation) != metadata["operation"]:
            raise ValueError("Loaded model differs from its checked checkpoint")
        expected = self.identity()
        def execute(request, context):
            # The backend must own this frozen loaded version, not reload a
            # mutable path or server's latest weights on an action. Full bytes
            # and provenance are audited when binding, not hashed per token.
            if self.identity() != expected or backend.identity() != metadata["model"]["backend"]:
                raise ValueError("Bound model implementation changed")
            return operation(request, context)
        return Operation(operation.name, value["revision"], operation.input, operation.output,
                         execute, operation.effects, (operation,))

    def install(self, parent, revision, output, *, candidate_id, binding_path="weights.json", **provenance):
        """Publish an owned candidate, changing only the declared weights component."""
        identifier(candidate_id)
        relative_path(binding_path)
        parent_sha = verify_harness(parent)
        self.read(revision)
        output = Path(output).resolve()
        if output.is_relative_to(Path(parent["root"]).resolve()) or output.is_relative_to(self.root):
            raise ValueError("Install checkpoint in a separate candidate workspace")
        request = dict(parent_sha256=parent_sha, checkpoint=revision, candidate_id=candidate_id,
                       binding_path=binding_path, provenance=provenance)
        with locked(output / ".install.lock"):
            from .self_harness.evaluation import freeze_json
            freeze_json(output / "request.json", request)
            target = output / "candidate"
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                child = read_json(manifest_path)
                binding = self.binding(child, binding_path=binding_path)
                if (binding["revision"] != revision or child.get("id") != candidate_id or child.get("root") != str(target)
                        or set(child) != {"id", "root", "components", *provenance}
                        or any(child.get(k) != v for k,v in provenance.items())
                        or any(child["components"][kind] != entries for kind,entries in parent["components"].items()
                               if kind != "weights")):
                    raise ValueError("Published checkpoint candidate differs from its installation request")
                return child
            temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=output))
            try:
                components = deepcopy(parent["components"])
                for kind, entries in components.items():
                    if kind == "weights":
                        continue
                    for name in entries:
                        destination = temporary / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(physical(Path(parent["root"]), name), destination)
                fixed = {name for kind, entries in components.items() if kind != "weights" for name in entries}
                if binding_path in fixed or any(name.startswith("model/") for name in fixed):
                    raise ValueError("Checkpoint files overlap fixed components")
                package = self.materialize(revision, temporary / "model")
                atomic_json(temporary / binding_path, dict(schema="physicalrsi.checkpoint-binding/v1",
                    store=self.identity(), checkpoint_revision=revision, package=str(package.relative_to(temporary))))
                components["weights"] = {str(p.relative_to(temporary)):file_digest(p)
                    for p in package.rglob("*") if p.is_file()}
                components["weights"][binding_path] = file_digest(temporary / binding_path)
                child = dict(provenance, id=candidate_id, root=str(temporary), components=components)
                self.binding(child, binding_path=binding_path)
                verify_harness(parent)
                for path in temporary.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                for path in [p for p in temporary.rglob("*") if p.is_dir()] + [temporary]:
                    HarnessSnapshots._sync(path)
                if target.exists():
                    # An interrupted publication may leave the exact owned tree.
                    if verify_harness(dict(child, root=str(target))) != verify_harness(child):
                        raise ValueError("Interrupted candidate publication changed")
                else:
                    os.rename(temporary, target)
                    HarnessSnapshots._sync(output)
                child["root"] = str(target)
                atomic_json(manifest_path, child)
                return child
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
