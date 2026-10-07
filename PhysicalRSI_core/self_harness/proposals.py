"""Declared JSON edits shared by reviewed search and external model proposals.

Strategies return data, never a workspace path or executable callback. The
materializer checks the edit domain; task admission still owns semantic safety,
dependency closure and isolation. Only independent trials determine selection.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from jsonschema import Draft202012Validator

from ..contracts import ReconciliationRequired
from ..infra.storage import atomic_json, canonical, digest, file_digest, locked, read_json, relative_path, strict_json
from .artifacts import CLOSURE, verify_harness
from .evaluation import freeze_json


def _path(name):
    if not isinstance(name, str) or relative_path(name).as_posix() != name:
        raise ValueError("Edit paths must be canonical relative paths")
    return name


def _physical(root, name):
    path = root / _path(name)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Proposal inputs must be physical files")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Proposal input escapes its root")
    return path


def _verify_materialized(child):
    verify_harness(child)
    root = Path(child["root"])
    expected = {name for entries in child["components"].values() for name in entries}
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Materialized candidates must contain physical files only")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError("Materialized candidate contains undeclared files")


@dataclass(frozen=True)
class ProposalLimits:
    max_candidates: int = 3
    request_bytes: int = 256 * 1024
    response_bytes: int = 256 * 1024
    closure_bytes: int = 64 * 1024 * 1024
    seconds: float = 120

    def __post_init__(self):
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 32:
            raise ValueError("Declare between one and 32 candidates")
        for value in (self.request_bytes, self.response_bytes, self.closure_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Positive proposal byte limits required")
        if type(self.seconds) not in (int, float) or not math.isfinite(self.seconds) or not 0 < self.seconds <= 600:
            raise ValueError("Proposal deadline must be finite and at most 600 seconds")


class JsonEditContract:
    """Map existing relative file paths to component names and local schemas."""

    def __init__(self, files):
        if not isinstance(files, dict) or not files:
            raise ValueError("Declare a nonempty editable file domain")
        self.files = deepcopy(files)
        for name, rule in self.files.items():
            _path(name)
            if (set(rule) != {"component", "schema"} or
                    rule["component"] not in CLOSURE - {"foundation"}):
                raise ValueError("Only declared non-foundation components are editable")
            self._local_schema(rule["schema"])
            Draft202012Validator.check_schema(rule["schema"])

    @classmethod
    def _local_schema(cls, value):
        # No resolver, remote reads or schema-provided executable behavior.
        if isinstance(value, dict):
            if {"$ref", "$dynamicRef", "$recursiveRef"}.intersection(value):
                raise ValueError("Edit schemas must be self-contained without references")
            for child in value.values():
                cls._local_schema(child)
        elif isinstance(value, list):
            for child in value:
                cls._local_schema(child)

    def public(self):
        return deepcopy(self.files)

    def inputs(self, parent, limits):
        verify_harness(parent)
        root = Path(parent["root"])
        owners, size = {}, 0
        for component, entries in parent["components"].items():
            for name in entries:
                owners.setdefault(_path(name), []).append(component)
        for name in owners:
            size += _physical(root, name).stat().st_size
        if size > limits.closure_bytes:
            raise ValueError("Declared harness exceeds proposal copy budget")
        result = {}
        for name, rule in self.files.items():
            if owners.get(name) != [rule["component"]]:
                raise ValueError("Editable file must belong to exactly its declared component")
            path = _physical(root, name)
            if path.stat().st_size > limits.request_bytes:
                raise ValueError("Editable input exceeds request budget")
            value = strict_json(path.read_text())
            # A seed must be in the same declared search domain as its children.
            if not Draft202012Validator(rule["schema"]).is_valid(value):
                raise ValueError("Parent is outside the declared edit schema")
            result[name] = dict(**deepcopy(rule), sha256=file_digest(path), value=value)
        return result

    def replacements(self, proposal, inputs):
        if not isinstance(proposal, dict) or set(proposal) != {"rationale", "edits"}:
            raise ValueError("Candidate must contain rationale and edits only")
        if not isinstance(proposal["rationale"], str) or not proposal["rationale"].strip():
            raise ValueError("Candidate rationale is required")
        edits = proposal["edits"]
        if not isinstance(edits, list) or not 1 <= len(edits) <= len(inputs):
            raise ValueError("Candidate must contain a bounded nonempty edit list")
        replacements = {}
        seen = set()
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"path", "before_sha256", "value"}:
                raise ValueError("Each edit requires path, before_sha256 and value only")
            name = _path(edit["path"])
            if name not in inputs or name in seen:
                raise ValueError("File is undeclared or edited more than once")
            seen.add(name)
            if edit["before_sha256"] != inputs[name]["sha256"]:
                raise ValueError("Edit is based on a different file revision")
            if not Draft202012Validator(inputs[name]["schema"]).is_valid(edit["value"]):
                raise ValueError("Edited value is outside its declared schema")
            if canonical(edit["value"]) != canonical(inputs[name]["value"]):
                replacements[name] = canonical(edit["value"]) + b"\n"
        return replacements


class EnumeratedEdits:
    """Reviewed finite search alternatives, passed through the same edit gate.

    This proposes declared values; it does not infer a repair or report success.
    The independent evaluator, rather than the order here, chooses the survivor.
    """

    def __init__(self, alternatives):
        if not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 32:
            raise ValueError("Declare between one and 32 search alternatives")
        self.alternatives = deepcopy(alternatives)
        canonical(self.alternatives)

    def identity(self):
        return dict(kind="enumerated-json-edits/v1", alternatives=deepcopy(self.alternatives),
                    implementation=file_digest(Path(__file__)))

    def generate(self, request, limits):
        candidates = []
        for alternative in self.alternatives:
            if not isinstance(alternative, dict) or not alternative or not set(alternative) <= set(request["editable"]):
                raise ValueError("Search alternative is outside the editable domain")
            if all(request["editable"][name]["value"] == value for name, value in alternative.items()):
                continue
            candidates.append(dict(rationale="Evaluate a declared search alternative", edits=[
                dict(path=name, before_sha256=request["editable"][name]["sha256"], value=value)
                for name, value in alternative.items()]))
        if len(candidates) > limits.max_candidates:
            raise ValueError("Search alternatives exceed candidate allowance")
        return dict(text=canonical(dict(parent_sha256=request["parent_sha256"], candidates=candidates)).decode(),
                    usage=None, provider=None, tool_calls=[])


class StructuredProposer:
    """One durable strategy call per round, then restartable data materialization.

    The strategy port is trusted host code with identity()/generate(). External
    model outputs are untrusted data. ModelProposal supplies an owned worker and
    total call deadline; arbitrary custom strategies must enforce their limits.
    """

    def __init__(self, *, evaluator, strategy, contract, objective, limits=None):
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Describe the task and proposed improvement objective")
        self.evaluator, self.strategy, self.contract = evaluator, strategy, contract
        self.objective, self.limits = objective, limits or ProposalLimits()
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind="structured-json-proposer/v1", strategy=self.strategy.identity(),
                    contract=self.contract.public(), objective=self.objective, limits=asdict(self.limits),
                    evaluator=self.evaluator.identity(), implementation=file_digest(Path(__file__)),
                    serialization=file_digest(Path(__file__).parents[1] / "infra/storage.py"))

    def _stable(self):
        if self.identity() != self._identity:
            raise ValueError("Frozen proposal configuration changed")

    def develop(self, parent, output):
        self._stable()
        return self.evaluator.development(parent, output)

    resume_development = develop

    def _request(self, parent, feedback, output):
        if (feedback.get("split") != "evolve" or not feedback.get("evidence") or
                not isinstance(feedback.get("episodes"), list) or not feedback["episodes"]):
            raise ValueError("Proposal requires development evidence and episode summaries")
        for name, sha in feedback["evidence"].items():
            if file_digest(_physical(output, name)) != sha:
                raise ValueError("Development evidence changed before proposal")
        rows = []
        for row in feedback["episodes"]:
            if set(row) != {"task", "case", "run_id", "outcome", "measurements"}:
                raise ValueError("Only declared development summaries may enter a proposal")
            if row["outcome"] not in {"success", "failure", "uncertain", "invalid"}:
                raise ValueError("Unknown development outcome")
            rows.append(deepcopy(row))
        request = dict(schema="physicalrsi.json-proposal-request/v1", objective=self.objective,
                       parent_sha256=verify_harness(parent), components=deepcopy(parent["components"]),
                       editable=self.contract.inputs(parent, self.limits),
                       development=dict(episodes=rows, evidence_sha256=sorted(feedback["evidence"].values())),
                       max_candidates=self.limits.max_candidates,
                       response_format=dict(parent_sha256="copy the request parent_sha256", candidates=[
                           dict(rationale="explain the proposed change", edits=[dict(
                               path="one declared editable path", before_sha256="its current sha256", value="new JSON value")])]))
        if len(canonical(request)) > self.limits.request_bytes:
            raise ValueError("Proposal request exceeds byte allowance")
        return request

    def propose(self, parent, feedback, output):
        output = Path(output).resolve()
        with locked(output / ".proposal.lock"):
            self._stable()
            request = self._request(parent, feedback, output)
            directory = output / "proposal"
            freeze_json(directory / "request.json", request)
            call_path = directory / "call.json"
            binding = dict(request_sha256=digest(request), proposer=self._identity)
            if call_path.exists():
                call = read_json(call_path)
                if call["binding"] != binding:
                    raise ValueError("Proposal call inputs changed")
                if call["state"] != "completed":
                    raise ReconciliationRequired("Proposal call outcome is unknown; automatic retry is forbidden")
                if digest(call["reply"]) != call["reply_sha256"]:
                    raise ValueError("Proposal response changed")
            else:
                if any((output / name).exists() for name in ("freeze.json", "validation-cases.json", "validation.json")):
                    raise ValueError("Cannot generate proposals after validation has been exposed")
                call = dict(schema="physicalrsi.proposal-call/v1", binding=binding,
                            state="started", reserved_calls=1)
                atomic_json(call_path, call)
                started = time.monotonic()
                try:
                    reply = self.strategy.generate(deepcopy(request), self.limits)
                    if len(canonical(reply)) > self.limits.response_bytes:
                        raise ValueError("Proposal response exceeds byte allowance")
                except BaseException as error:
                    # Exception messages may contain provider URLs or secrets.
                    atomic_json(call_path, dict(call, error_type=type(error).__name__))
                    raise ReconciliationRequired("Proposal call has no durable response; reconciliation required") from None
                call = dict(call, state="completed", reply=reply, reply_sha256=digest(reply),
                            elapsed_seconds=time.monotonic() - started)
                atomic_json(call_path, call)
            self._stable()
            if verify_harness(parent) != request["parent_sha256"]:
                raise ValueError("Parent changed during proposal")
            return self._materialize(parent, feedback, output, request, call)

    resume_proposal = propose

    def _materialize(self, parent, feedback, output, request, call):
        directory = output / "proposal"
        report_path = directory / "materialization.json"
        if report_path.exists():
            report = read_json(report_path)
            if (report["call_sha256"] != file_digest(directory / "call.json") or
                    digest(report["result"]) != report["result_sha256"]):
                raise ValueError("Proposal materialization receipt changed")
            for child in report["result"]["candidates"]:
                _verify_materialized(child)
            return deepcopy(report["result"]["candidates"])
        children, rejected = [], []
        try:
            reply = call["reply"]
            if not isinstance(reply, dict) or reply.get("tool_calls"):
                raise ValueError("Proposal tools are not permitted")
            batch = strict_json(reply["text"])
            if (not isinstance(batch, dict) or set(batch) != {"parent_sha256", "candidates"}
                    or batch["parent_sha256"] != request["parent_sha256"]
                    or not isinstance(batch["candidates"], list)
                    or len(batch["candidates"]) > self.limits.max_candidates):
                raise ValueError("Response does not match the frozen proposal request")
        except (ValueError, KeyError, TypeError, RecursionError):
            batch = dict(candidates=[])
            rejected.append(dict(index=None, reason="invalid_response"))
        seen = {request["parent_sha256"]}
        evidence = dict(feedback["evidence"], **{
            str(p.relative_to(output)): file_digest(p) for p in (directory / "request.json", directory / "call.json")})
        for index, proposal in enumerate(batch["candidates"]):
            try:
                replacements = self.contract.replacements(proposal, request["editable"])
            except (ValueError, KeyError, TypeError):
                rejected.append(dict(index=index, reason="edit_contract_rejected"))
                continue
            files = {name for entries in parent["components"].values() for name in entries}
            size = sum(len(replacements[name]) if name in replacements else
                       _physical(Path(parent["root"]), name).stat().st_size for name in files)
            if size > self.limits.closure_bytes:
                rejected.append(dict(index=index, reason="candidate_copy_budget_exceeded"))
                continue
            components = deepcopy(parent["components"])
            for name, payload in replacements.items():
                components[request["editable"][name]["component"]][name] = hashlib.sha256(payload).hexdigest()
            sha = digest(dict(components=components))
            if sha in seen:
                rejected.append(dict(index=index, reason="unchanged_or_duplicate"))
                continue
            seen.add(sha)
            name = "proposal-" + sha
            root = directory / "candidates" / name
            child = dict(id=name, root=str(root), components=components, parent_sha256=request["parent_sha256"],
                         method=dict(strategy=self.strategy.identity(), rationale=proposal["rationale"]),
                         changes=deepcopy(proposal["edits"]), environment=self.evaluator.identity(), evidence=evidence,
                         costs=dict(reserved_calls=1, call_receipt="proposal/call.json", shared_batch=True,
                                    provider_reported_usage=deepcopy(call["reply"].get("usage"))))
            if not root.exists():
                root.parent.mkdir(parents=True, exist_ok=True)
                # Only declared files are copied. No parent workspace, cache,
                # credentials, hard links or undeclared dependency is inherited.
                with tempfile.TemporaryDirectory(prefix=".materialize-", dir=root.parent) as staging:
                    temporary = Path(staging) / "harness"
                    for entries in components.values():
                        for relative in entries:
                            destination = temporary / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if relative in replacements:
                                destination.write_bytes(replacements[relative])
                            else:
                                shutil.copyfile(_physical(Path(parent["root"]), relative), destination)
                    _verify_materialized(dict(child, root=str(temporary)))
                    temporary.rename(root)
            _verify_materialized(child)
            children.append(child)
        if verify_harness(parent) != request["parent_sha256"]:
            raise ValueError("Parent changed during materialization")
        result = dict(candidates=children, rejected=rejected, qualification=None)
        atomic_json(report_path, dict(call_sha256=file_digest(directory / "call.json"),
                                      result=result, result_sha256=digest(result)))
        return deepcopy(children)
