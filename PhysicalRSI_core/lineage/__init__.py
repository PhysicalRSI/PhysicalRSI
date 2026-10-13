"""Local CAS lineage for a scoped harness, with immutable history and rollback.

This records assessed selections, not a physical capability certificate. The caller owns
independent evaluation; one host with working POSIX flock owns this store.
"""

from copy import deepcopy
from pathlib import Path

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    locked,
    read_json,
)
from PhysicalRSI_core.self_harness.artifacts import EvolutionSpec, HarnessSnapshots, _HarnessReadAudit, harness_identity, verify_harness


class StateConflict(ValueError):
    pass


class HarnessState:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.snapshots = HarnessSnapshots(self.root / "snapshots")

    def _metadata(self, revision):
        if (not isinstance(revision, str) or len(revision) != 64
                or any(c not in "0123456789abcdef" for c in revision)):
            raise ValueError("Expected a content-addressed lineage revision")
        value = read_json(self.root / "history" / (revision + ".json"))
        if digest(value) != revision:
            raise ValueError("Lineage record changed")
        return value

    def _record(self, revision):
        return self._audit_record(revision, _HarnessReadAudit(), {})

    def _audit_record(self, revision, harnesses, evidence):
        value = self._metadata(revision)
        if harnesses.verify(value["harness"]) != value["freeze_sha256"]:
            raise ValueError("Committed harness changed")
        for path, sha in value.get("evidence", {}).items():
            path = Path(path)
            if path not in evidence:
                evidence[path] = file_digest(path)
            if evidence[path] != sha:
                raise ValueError("Committed evidence changed")
        return value

    def resolve(self):
        revision = read_json(self.root / "current.json")["revision"]
        return self.read(revision)

    def read(self, revision):
        """Inspect a historical revision and verify its complete evidence."""
        record = self._record(revision)
        return dict(revision=revision, **record)

    def history(self, revision):
        """Audit a complete ancestry, newest first, sharing checks within this read.

        Repeated snapshots and evidence paths are read once. Every declaration
        is still compared with the checked hash; the next call reads them again.
        """
        if revision is None:
            raise ValueError("Expected a content-addressed lineage revision")
        records, seen, evidence, harnesses = [], set(), {}, _HarnessReadAudit()
        while revision is not None:
            record = self._audit_record(revision, harnesses, evidence)
            if revision in seen:
                raise ValueError("Cyclic lineage history")
            seen.add(revision)
            records.append(dict(revision=revision, **record))
            revision = record["previous"]
        return records

    def initialize(self, initial, *, policy, scope):
        with locked(self.root / ".state.lock"):
            if (self.root / "current.json").exists():
                current = self.resolve()
                if current["policy"] != policy or current["scope"] != scope:
                    raise ValueError("State scope or frozen policy changed")
                return current
            body = dict(
                harness=self.snapshots.capture(initial),
                freeze_sha256=verify_harness(initial),
                policy=policy,
                scope=scope,
                previous=None,
                action="bootstrap",
                evidence={},
            )
            return self._publish(body)

    def _publish(self, body):
        revision = digest(body)
        path = self.root / "history" / (revision + ".json")
        if path.exists() and read_json(path) != body:
            raise ValueError("Conflicting lineage history")
        atomic_json(path, body)
        atomic_json(self.root / "current.json", dict(revision=revision))
        return dict(revision=revision, **deepcopy(body))

    def commit(
        self,
        expected,
        survivor,
        *,
        evidence,
        decision,
        selector=None,
        comparison_root=None,
    ):
        with locked(self.root / ".state.lock"):
            current = self.resolve()
            freeze = verify_harness(survivor)
            # Idempotent after a crash between CAS and the loop's commit receipt.
            if (
                current["previous"] == expected
                and current.get("decision") == decision
                and current["evidence"] == evidence
                and current["freeze_sha256"] == freeze
            ):
                return current
            if current["revision"] != expected:
                raise StateConflict(
                    "Committed parent changed; preserve this comparison"
                )
            if selector is None or comparison_root is None:
                raise ValueError(
                    "Commit requires the frozen selector and full comparison"
                )
            from PhysicalRSI_core.self_harness.bindings import policy_identity

            research = current["policy"].get("schema") == "physicalrsi.campaign-policy/v2"
            selector_identity = policy_identity(selector) if research else selector.identity()
            if selector_identity != current["policy"]["identities"]["selector"]:
                raise ValueError("Selection policy changed")
            directory = Path(comparison_root).resolve()

            def completed(name):
                path = directory / (name + ".json")
                receipt = read_json(path)
                if (
                    evidence.get(str(path)) != file_digest(path)
                    or receipt.get("state") != "completed"
                    or digest(receipt["output"]) != receipt["output_sha256"]
                ):
                    raise ValueError("Missing or changed comparison receipt")
                return receipt["output"]

            comparison = completed("freeze")
            policy = current["policy"]
            round_protocol = policy["protocol"]
            if policy.get("schema") == "physicalrsi.campaign-policy/v2":
                config_path = directory / "config.json"
                config = read_json(config_path)
                if (evidence.get(str(config_path)) != file_digest(config_path)
                        or config.get("campaign_policy") != policy
                        or comparison.get("round_config_sha256") != digest(config)
                        or config["identities"]["selector"] != selector.identity()
                        or config["round_context"]["parent_revision"] != expected):
                    raise ValueError("Round configuration is not bound to campaign policy and parent")
                round_protocol = config["protocol"]
            if (
                comparison["profile"] != current["policy"]["profile"]
                or comparison["protocol_sha256"]
                != digest(round_protocol)
                or comparison["scope"] != current["scope"]
                or comparison["candidates"].get(current["harness"]["id"])
                != current["freeze_sha256"]
            ):
                raise ValueError(
                    "Comparison policy or parent differs from committed state"
                )
            cohort = completed("validation")
            results = []
            for name, sha in comparison["candidates"].items():
                admission = completed("admit_" + name)
                if (
                    admission.get("accepted") is not True
                    or admission.get("freeze_sha256") != sha
                ):
                    raise ValueError("Candidate not admitted")
                result = completed("evaluate_" + name)
                if (
                    result.get("evaluator_revision")
                    != round_protocol["identity"]
                ):
                    raise ValueError("Evaluator protocol changed")
                results.append(result)
            if (
                completed("selection") != decision
                or selector.select(comparison, cohort, results, evidence_root=directory)
                != decision
            ):
                raise ValueError("Decision differs from full paired evidence")
            if decision["parent_id"] != current["harness"]["id"]:
                raise StateConflict("Decision does not compare the expected parent")
            if (
                decision["survivor_id"] != survivor["id"]
                or decision["survivor_freeze_sha256"] != freeze
            ):
                raise ValueError("Decision and survivor differ")
            if (
                survivor["components"]["foundation"]
                != current["harness"]["components"]["foundation"]
            ):
                raise ValueError("Foundation F changed")
            if ('file_modes' in survivor) != ('file_modes' in current['harness']):
                raise ValueError('Selection cannot add or drop file_modes')
            if any(survivor.get('file_modes', {}).get(n) != current['harness'].get('file_modes', {}).get(n)
                   for n in survivor['components']['foundation']):
                raise ValueError('Foundation file mode changed')
            if "evolution" in current["policy"]:
                EvolutionSpec.from_identity(current["policy"]["evolution"]).compare(current["harness"], survivor)
            if not evidence:
                raise ValueError("Commit requires comparison evidence")
            for path, sha in evidence.items():
                if file_digest(Path(path)) != sha:
                    raise ValueError("Comparison evidence changed before commit")
            if decision["decision"] == "retain_parent":
                if freeze != current["freeze_sha256"]:
                    raise ValueError("Retain must preserve the parent")
                action = "assessment"
            elif decision["decision"] == "inherit_child":
                action = "selection"
            else:
                raise ValueError("Unknown decision")
            body = dict(
                harness=deepcopy(current["harness"]) if action == "assessment" else self.snapshots.capture(survivor),
                freeze_sha256=freeze,
                policy=current["policy"],
                scope=current["scope"],
                previous=expected,
                action=action,
                evidence=deepcopy(evidence),
                decision=deepcopy(decision),
            )
            result = self._publish(body)
            self._record(result["revision"])
            return result

    def rollback(self, expected, *, target=None, reason="operator rollback"):
        """CAS to an audited ancestor, even if the active artifact is damaged.

        Current and intervening history metadata must still hash correctly. The
        target's files and evidence must pass full verification. Unknown history
        corruption requires separate recovery; rollback never bypasses it.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Record a rollback reason")
        with locked(self.root / ".state.lock"):
            revision = read_json(self.root / "current.json")["revision"]
            current = self._metadata(revision)

            def previous_selection(record):
                # Assessments advance evidence, not the deployed artifact.
                # Consecutive rollbacks step back selected generations.
                while record["action"] in {"rollback", "assessment"}:
                    record = self._metadata(record["rollback_to"] if record["action"] == "rollback"
                                            else record["previous"])
                return record["previous"]

            if (current["action"] == "rollback" and current["previous"] == expected
                    and current.get("rollback_reason") == reason):
                requested = previous_selection(self._metadata(expected)) if target is None else target
                if current["rollback_to"] == requested:
                    return self.read(revision)
            if revision != expected:
                raise StateConflict("Committed parent changed before rollback")
            if target is None:
                target = previous_selection(current)
            if target is None:
                raise ValueError("No previous harness")
            ancestor, seen = current["previous"], set()
            while ancestor is not None and ancestor != target:
                if ancestor in seen:
                    raise ValueError("Cyclic lineage history")
                seen.add(ancestor)
                ancestor = self._metadata(ancestor)["previous"]
            if ancestor is None:
                raise ValueError("Rollback target is not an ancestor")
            previous = self._record(target)
            if previous["policy"] != current["policy"] or previous["scope"] != current["scope"]:
                raise ValueError("Rollback target has another policy or scope")
            return self._publish(
                dict(
                    previous,
                    previous=expected,
                    action="rollback",
                    rollback_to=target,
                    rollback_reason=reason,
                )
            )

    def restore_snapshot(self, expected, *, revision, source, reason):
        """Restore an owned ancestor/current snapshot from a verified replica.

        This explicit maintenance operation preserves the lineage head and the
        original artifact hash. It does not resume workers or change selection.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Record a snapshot restoration reason")
        with locked(self.root / ".state.lock"):
            head = read_json(self.root / "current.json")["revision"]
            if head != expected:
                raise StateConflict("Committed parent changed before snapshot restoration")
            current, seen = head, set()
            while current != revision:
                if current is None:
                    raise ValueError("Snapshot restoration target is not an ancestor")
                if current in seen:
                    raise ValueError("Cyclic lineage history")
                seen.add(current)
                current = self._metadata(current)["previous"]
            record = self._metadata(revision)
            if digest(harness_identity(record["harness"])) != record["freeze_sha256"]:
                raise ValueError("Snapshot restoration declaration changed")
            return self.snapshots.restore(record["harness"], source,
                operation=dict(expected_head=expected, target_revision=revision, reason=reason,
                               implementation=file_digest(Path(__file__))))

    def reserve_validation(self, comparison, layouts):
        with locked(self.root / ".validation.lock"):
            path = self.root / "validation.json"
            values = read_json(path) if path.exists() else {}
            for key, used in values.items():
                if key != comparison and set(used).intersection(layouts):
                    raise ValueError("Validation inputs reused across interactions")
            if comparison in values and values[comparison] != layouts:
                raise ValueError("Validation inputs changed")
            values[comparison] = layouts
            atomic_json(path, values)

    def reserve_splits(self, round_identity, splits):
        """Register semantic reset IDs before development, across all rounds.

        Development may repeat development; test may repeat test. Validation
        never overlaps another round or another split, including metadata aliases.
        """
        if set(splits) != {"development", "validation", "test"}:
            raise ValueError("Declare all research splits")
        sets = {key: set(values) for key, values in splits.items()}
        if any(not values or len(values) != len(splits[key]) for key, values in sets.items()):
            raise ValueError("Empty or duplicate split identities")
        if any(sets[a] & sets[b] for a, b in (("development", "validation"), ("development", "test"),
                                             ("validation", "test"))):
            raise ValueError("Research splits overlap")
        with locked(self.root / ".splits.lock"):
            path = self.root / "splits.json"
            ledger = read_json(path) if path.exists() else {}
            if round_identity in ledger and ledger[round_identity] != splits:
                raise ValueError("Frozen round splits changed")
            for key, prior in ledger.items():
                if key == round_identity:
                    continue
                for split, values in sets.items():
                    forbidden = set().union(*(set(v) for name, v in prior.items()
                                              if split == "validation" or name != split))
                    if values & forbidden:
                        raise ValueError("Reset identity reused across research rounds or splits")
            ledger[round_identity] = deepcopy(splits)
            atomic_json(path, ledger)
