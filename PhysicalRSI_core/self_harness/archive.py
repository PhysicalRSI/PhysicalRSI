"""Owned candidate branches and historical evidence for evolutionary search.

Entries are immutable completed-round observations, not deployment authority.
Reused branches must pass fresh admission and paired validation against the
current incumbent. Local POSIX storage assumes cooperating, trusted adapters.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json, relative_path
from ..infra import owned_files
from . import artifacts
from .artifacts import EvolutionSpec, HarnessSnapshots, _HarnessReadAudit, verify_harness
from .bindings import policy_identity, validate_admission
from .protocol import _sha


def _physical(root, name):
    path = root / relative_path(name)
    if str(path.relative_to(root)) != name:
        raise ValueError("Archive evidence path is not canonical")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError("Archive evidence must be physical files")
    if not path.is_file():
        raise ValueError("Archive evidence is missing: " + name)
    return path


class CandidateArchive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.snapshots = HarnessSnapshots(self.root / "harnesses")

    def identity(self):
        from ..experiments import runtime_implementation
        return dict(schema="physicalrsi.candidate-archive/v1", root=str(self.root),
                    implementation=file_digest(Path(__file__)),
                    dependencies=dict(artifacts=file_digest(Path(artifacts.__file__)),
                                      owned_files=file_digest(Path(owned_files.__file__)),
                                      development_reader=runtime_implementation()))

    def capture_round(self, directory, *, state, selector):
        """Publish a completed round after rechecking selection and lineage.

        Retain declared development/validation evidence, including binary
        media. Do not copy preregistered test cases or final-test reports.
        No environment, proposer or evaluator execution occurs here.
        """
        directory = Path(directory).resolve()
        if self.root.is_relative_to(directory) or directory.is_relative_to(self.root):
            raise ValueError("Keep the candidate archive outside round evidence")
        files = {}

        def include(name, expected=None):
            path = _physical(directory, name)
            sha = file_digest(path)
            if expected is not None and sha != expected:
                raise ValueError("Round evidence changed before archive capture")
            if name in files and files[name] != sha:
                raise ValueError("Round changed during archive capture")
            files[name] = sha
            return path

        def document(name):
            path = include(name)
            value = read_json(path)
            if file_digest(path) != files[name]:
                raise ValueError("Round document changed during archive capture")
            return value

        def completed(name):
            value = document(name + ".json")
            if value.get("state") != "completed" or digest(value["output"]) != value["output_sha256"]:
                raise ValueError("Archive requires completed stage receipts: " + name)
            return value["output"]

        config, anchor = document("config.json"), document("parent.json")
        view = document("archive-view.json") if (directory / "archive-view.json").exists() else None
        policy = config.get("campaign_policy", config)
        if view is not None:
            self.summaries(view)
            if view["scope"] != config["scope"] or view["policy_sha256"] != digest(policy):
                raise ValueError("Archive search history differs from the completed round")
        if (state.read(anchor["revision"]) != anchor or anchor["policy"] != policy
                or anchor["scope"] != config["scope"]):
            raise ValueError("Archive parent differs from audited lineage")
        expected_selector = (policy_identity(selector) if "campaign_policy" in config else selector.identity())
        if expected_selector != policy["identities"]["selector"]:
            raise ValueError("Archive selector differs from the frozen policy")
        development, children = completed("develop"), completed("propose")
        if development.get("split") != "evolve" or not development.get("evidence"):
            raise ValueError("Archive requires development evidence")
        for name, sha in development["evidence"].items():
            if relative_path(name).parts[:2] != ("experiments", "development"):
                raise ValueError("Archive development evidence crossed its split")
            include(name, sha)
        pool = [anchor["harness"], *children]
        hashes = {h["id"]: verify_harness(h) for h in pool}
        if len(hashes) != len(pool) or len(set(hashes.values())) != len(pool):
            raise ValueError("Archive candidate pool has duplicate identities")
        evolution = EvolutionSpec.from_identity(config["evolution"])
        for child in children:
            evolution.compare(pool[0], child)
            if child.get("parent_sha256") != anchor["freeze_sha256"]:
                raise ValueError("Archived child belongs to another parent")
            if "archive_origin" in child:
                if view is None:
                    raise ValueError("Archived branch requires a frozen search view")
                self.verify_origin(view, child["archive_origin"])
        admissions, results, comparison, cohort, decision, screening = {}, [], None, None, None, None
        if children:
            for h in pool:
                admissions[h["id"]] = validate_admission(completed("admit_" + h["id"]), hashes[h["id"]])
            if (directory / "screening.json").exists():
                screening = completed("screening")
                if (screening.get("role") != "screening_only" or not screening.get("evidence")
                        or set(screening["candidates"]) != {name for name, row in admissions.items() if row["accepted"]}):
                    raise ValueError("Archived screening differs from admitted candidates")
                for path, sha in screening["evidence"].items():
                    if relative_path(path).parts[:2] != ("experiments", "screening"):
                        raise ValueError("Archive screening evidence crossed its stage")
                    include(path, sha)
                for name, row in screening["candidates"].items():
                    screened = validate_admission(dict(row, evidence=screening["evidence"]), hashes[name])
                    if not screened["accepted"]:
                        if name == pool[0]["id"]:
                            raise ValueError("Archived screening removed the incumbent")
                        admissions[name] = screened
            admitted = completed("admission")
            if set(admitted["accepted"]) != {name for name, row in admissions.items() if row["accepted"]}:
                raise ValueError("Archived final admission differs from its stages")
        if (directory / "selection.json").exists():
            comparison, cohort = completed("freeze"), completed("validation")
            accepted = {name: sha for name, sha in hashes.items() if admissions[name]["accepted"]}
            if (comparison["candidates"] != accepted or comparison["parent_id"] != pool[0]["id"]
                    or comparison["profile"] != config["profile"]
                    or comparison["protocol_sha256"] != digest(config["protocol"])
                    or comparison["scope"] != config["scope"]):
                raise ValueError("Archived comparison differs from the frozen round")
            for name in accepted:
                result = completed("evaluate_" + name)
                if result["evaluator_revision"] != config["protocol"]["identity"]:
                    raise ValueError("Archived evaluator revision changed")
                for episode in result["episodes"]:
                    for path, sha in episode["evidence_sha256"].items():
                        if relative_path(path).parts[:2] != ("experiments", "validation"):
                            raise ValueError("Archive validation evidence crossed its split")
                        include(path, sha)
                results.append(result)
            if (directory / "evaluation_pool.json").exists():
                pooled = completed("evaluation_pool")
                if sorted(pooled, key=lambda r: r["candidate_id"]) != sorted(results, key=lambda r: r["candidate_id"]):
                    raise ValueError("Archived pooled evaluation differs from individual results")
            decision = completed("selection")
            if selector.select(comparison, cohort, results, evidence_root=directory) != decision:
                raise ValueError("Archive decision differs from verified evidence")
            committed = document("commit.json")
            if state.read(committed["revision"]) != committed:
                raise ValueError("Archive result differs from audited lineage")
            if (committed["freeze_sha256"] != decision["survivor_freeze_sha256"]
                    or committed["previous"] != anchor["revision"] or committed.get("decision") != decision
                    or committed["action"] != ("assessment" if decision["decision"] == "retain_parent" else "selection")
                    or (decision["decision"] == "retain_parent" and
                        committed["harness"] != anchor["harness"])):
                raise ValueError("Archive selection was not committed against this parent")
            outcome = decision["decision"]
        else:
            outcome = document("retained.json")["reason"]
            expected = "all_candidates_rejected" if children else "no_changed_candidate"
            if (outcome != expected or any(row["accepted"] for name, row in admissions.items()
                                         if name != pool[0]["id"])):
                raise ValueError("Round is incomplete, not a retained comparison")
            committed = anchor
        candidates = {}
        for h in pool:
            name = h["id"]
            selected = hashes[name] == committed["freeze_sha256"]
            admission = admissions.get(name)
            status = ("selected" if selected else "admission_rejected" if admission and not admission["accepted"]
                      else "evaluated_not_selected")
            candidates[name] = dict(harness=self.snapshots.capture(h), freeze_sha256=hashes[name],
                disposition=status, admission=admission,
                screening=deepcopy(screening["candidates"].get(name)) if screening else None,
                metrics=deepcopy(decision["metrics"].get(name)) if decision else None,
                rejection=deepcopy(decision["rejected"].get(name)) if decision else None)
        body = dict(schema="physicalrsi.archive-round/v1", source_round=str(directory),
                    policy=policy, policy_sha256=digest(policy), scope=config["scope"],
                    parent_revision=anchor["revision"], result_revision=committed["revision"],
                    outcome=outcome, candidates=candidates, comparison=comparison, cohort=cohort,
                    results=results, decision=decision, files=files, qualification=None)
        revision = digest(body)
        with locked(self.root / ".archive.lock"):
            entries = self.root / "entries"
            entries.mkdir(exist_ok=True)
            target = entries / revision
            if not target.exists():
                temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=entries))
                try:
                    for name, sha in sorted(files.items()):
                        destination = temporary / "evidence" / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with _physical(directory, name).open("rb") as incoming, destination.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                        if file_digest(destination) != sha:
                            raise ValueError("Evidence changed while archiving")
                    if decision and selector.select(comparison, cohort, results,
                                                    evidence_root=temporary / "evidence") != decision:
                        raise ValueError("Copied evidence does not reproduce selection")
                    atomic_json(temporary / "record.json", body)
                    for folder in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                                         key=lambda p: len(p.parts), reverse=True):
                        HarnessSnapshots._sync(folder)
                    HarnessSnapshots._sync(temporary)
                    os.rename(temporary, target)
                    HarnessSnapshots._sync(entries)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            result = self.read(revision)
            if result != dict(revision=revision, **body):
                raise ValueError("Conflicting archived round")
            return result

    def read(self, revision):
        """Verify an entry and the historical views on which it depended."""
        return self._read_many([revision])[revision]

    def _read_many(self, revisions):
        # Iterative traversal avoids recursion limits on long campaigns and
        # hashes each owned entry only once per read, even with shared history.
        records, views, pending = {}, {}, list(revisions)
        harnesses = _HarnessReadAudit()
        while pending:
            revision = pending.pop()
            if revision in records:
                continue
            record, view = self._read_one(revision, harnesses)
            records[revision] = record
            if view is None:
                continue
            self._validate_view(view)
            if view["scope"] != record["scope"] or view["policy_sha256"] != record["policy_sha256"]:
                raise ValueError("Archived search history crossed its scope or policy")
            views[revision] = view
            pending.extend(r for r in view["revisions"] if r not in records)
        for view in views.values():
            for revision in view["revisions"]:
                self._verify_scope(records[revision], view)
        return records

    def _read_one(self, revision, harnesses):
        folder = self.root / "entries" / _sha(revision)
        payload = _physical(folder, "record.json").read_bytes()
        value = json.loads(payload)
        if digest(value) != revision or value.get("schema") != "physicalrsi.archive-round/v1":
            raise ValueError("Archived record changed")
        for row in value["candidates"].values():
            if (Path(row["harness"]["root"]) != self.snapshots.root / row["freeze_sha256"]
                    or harnesses.verify(row["harness"]) != row["freeze_sha256"]):
                raise ValueError("Archived candidate changed or escaped its snapshot")
        evidence = folder / "evidence"
        actual = owned_files._file_names(evidence)
        if actual != set(value["files"]):
            raise ValueError("Archived evidence closure changed")
        view = None
        for name, sha in value["files"].items():
            path = _physical(evidence, name)
            if name == "archive-view.json":
                checked = path.read_bytes()
                actual_sha = sha256(checked).hexdigest()
            else:
                actual_sha = file_digest(path)
            if actual_sha != sha:
                raise ValueError("Archived evidence changed")
            if name == "archive-view.json":
                view = json.loads(checked)
                if not isinstance(view, dict):
                    raise ValueError("Invalid archived search view")
        if file_digest(_physical(folder, "record.json")) != sha256(payload).hexdigest():
            raise ValueError("Archived record changed during audit")
        return dict(revision=revision, **value), view

    def view(self, *, scope, policy_sha256):
        """Freeze a search view. Later appends do not enter a resumed proposal."""
        _sha(policy_sha256)
        if not isinstance(scope, str) or not scope:
            raise ValueError("Archive search requires an explicit scope")
        entries = sorted(path.parent.name for path in (self.root / "entries").glob("*/record.json")
                         if not path.parent.name.startswith(".pending-"))
        records = self._read_many(entries)
        revisions = [revision for revision in entries
                     if records[revision]["scope"] == scope and records[revision]["policy_sha256"] == policy_sha256]
        return dict(schema="physicalrsi.archive-view/v1", archive=self.identity(), scope=scope,
                    policy_sha256=policy_sha256, revisions=revisions)

    def _validate_view(self, view):
        if (view.get("schema") != "physicalrsi.archive-view/v1" or view["archive"] != self.identity()
                or view["revisions"] != sorted(set(view["revisions"]))):
            raise ValueError("Invalid or changed archive view")
        if not isinstance(view["scope"], str) or not view["scope"]:
            raise ValueError("Archive search requires an explicit scope")
        _sha(view["policy_sha256"])
        for revision in view["revisions"]:
            _sha(revision)

    @staticmethod
    def _verify_scope(record, view):
        if record["scope"] != view["scope"] or record["policy_sha256"] != view["policy_sha256"]:
            raise ValueError("Archive view crossed its scope or search policy")

    def _read_view(self, view):
        self._validate_view(view)
        records = self._read_many(view["revisions"])
        for revision in view["revisions"]:
            self._verify_scope(records[revision], view)
        return records

    def summaries(self, view):
        records = self._read_view(view)
        rows = []
        for revision in view["revisions"]:
            record = records[revision]
            for name, value in sorted(record["candidates"].items()):
                rows.append(dict(archive_revision=revision, candidate_id=name,
                    freeze_sha256=value["freeze_sha256"], disposition=value["disposition"],
                    admission=deepcopy(value["admission"]), metrics=deepcopy(value["metrics"]),
                    screening=deepcopy(value.get("screening")),
                    rejection=deepcopy(value["rejection"]), parent_revision=record["parent_revision"]))
        return rows

    def development(self, view):
        """Read owned common-Trial development and screening observations.

        Validation scores and final reports are not training observations here.
        Legacy archives without common Trial receipts yield no such observations.
        """
        from ..experiments import ExperimentRuntime
        from ..trials import Evaluation
        records = self._read_view(view)
        observations = []
        for revision in view["revisions"]:
            record = records[revision]
            root = self.root / "entries" / revision
            evidence = root / "evidence"
            for name in sorted(record["files"]):
                parts = relative_path(name).parts
                if (len(parts) == 6 and parts[:2] == ("experiments", "development")
                        and parts[3] == "trials" and parts[-1] == "receipt.json"):
                    role, candidate_id = "incumbent_development", parts[2]
                elif (len(parts) == 9 and parts[:5] ==
                      ("experiments", "screening", "development", "experiments", "development")
                      and parts[6] == "trials" and parts[-1] == "receipt.json"):
                    role, candidate_id = "development_screening", parts[5]
                else:
                    continue
                path = evidence / name
                value = Evaluation.read(ExperimentRuntime(path.parent.parent), path.parent.name).to_dict()
                origin = value["trial"]["system2"]
                candidate = record["candidates"].get(candidate_id)
                if (candidate is None or origin is None or origin["split"] != "development"
                        or origin["candidate_id"] != candidate_id
                        or origin["candidate_sha256"] != candidate["freeze_sha256"]
                        or value["trial"]["scope"] != record["scope"]):
                    raise ValueError("Archived development Trial differs from its candidate or scope")
                for raw_path, sha in value["evidence"].items():
                    relative = str(Path(raw_path).relative_to(evidence))
                    if record["files"].get(relative) != sha:
                        raise ValueError("Archived development evidence is outside its owned closure")
                observations.append(dict(archive_revision=revision, role=role,
                    candidate=deepcopy(candidate["harness"]), evaluation=value,
                    archive_evidence={str(root / "record.json"): file_digest(root / "record.json")}))
            payload = _physical(root, "record.json").read_bytes()
            if digest(json.loads(payload)) != revision:
                raise ValueError("Archived record changed during development read")
            marker_sha = sha256(payload).hexdigest()
            if any(row["archive_revision"] == revision and
                   row["archive_evidence"][str(root / "record.json")] != marker_sha for row in observations):
                raise ValueError("Archived record changed during development read")
        return observations

    def verify_origin(self, view, origin):
        rows = self.summaries(view)
        if (origin.get("view_sha256") != digest(view) or not any(
                row["archive_revision"] == origin.get("revision")
                and row["candidate_id"] == origin.get("candidate_id")
                and row["freeze_sha256"] == origin.get("freeze_sha256") for row in rows)):
            raise ValueError("Candidate origin is outside the frozen archive view")

    def fork(self, view, revision, candidate_id, *, parent, output, name, evolution):
        """Materialize a branch proposal; never update the committed head."""
        records = self._read_view(view)
        if revision not in view["revisions"]:
            raise ValueError("Branch is outside the frozen archive view")
        record = records[revision]
        row = record["candidates"][candidate_id]
        source = row["harness"]
        changes = evolution.compare(parent, source)
        if not changes:
            raise ValueError("Archived branch has the current executable content")
        identifier(name)
        if name == parent["id"]:
            raise ValueError("Branch proposal needs a distinct ID")
        if Path(output).is_symlink():
            raise ValueError("Branch output cannot be a symlink")
        output = Path(output).resolve()
        if output.exists() or output.is_relative_to(self.root) or self.root.is_relative_to(output):
            raise ValueError("Materialize a branch in a new directory outside the archive")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".branch-", dir=output.parent))
        try:
            for relative in {p for files in source["components"].values() for p in files}:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(source["root"]) / relative, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
            candidate = dict(id=name, root=str(temporary), components=deepcopy(source["components"]))
            if verify_harness(candidate) != row["freeze_sha256"]:
                raise ValueError("Archived branch changed during materialization")
            for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                HarnessSnapshots._sync(directory)
            HarnessSnapshots._sync(temporary)
            os.rename(temporary, output)
            HarnessSnapshots._sync(output.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        folder = self.root / "entries" / revision
        evidence = {str(folder / "record.json"): file_digest(folder / "record.json")}
        evidence.update({str(folder / "evidence" / p): sha for p, sha in record["files"].items()})
        evidence.update({str(Path(source["root"]) / p): sha
                         for files in source["components"].values() for p, sha in files.items()})
        return dict(candidate, root=str(output), parent_sha256=verify_harness(parent),
                    method="archive-branch-reuse/v1", changes=changes, environment=dict(scope=record["scope"]),
                    evidence=evidence, costs=dict(archive_materializations=1), used_lessons=[],
                    archive_origin=dict(revision=revision, candidate_id=candidate_id,
                                        freeze_sha256=row["freeze_sha256"], view_sha256=digest(view)))


def verify_archive_receipt(directory):
    """Audit optional archive ownership when a campaign reopens old rounds."""
    path = Path(directory) / "archive.json"
    if path.exists():
        receipt = read_json(path)
        archive = CandidateArchive(receipt["archive"]["root"])
        if archive.identity() != receipt["archive"]:
            raise ValueError("Candidate archive implementation changed")
        record = archive.read(receipt["revision"])
        if record["source_round"] != str(Path(directory).resolve()):
            raise ValueError("Archive receipt belongs to another round")
