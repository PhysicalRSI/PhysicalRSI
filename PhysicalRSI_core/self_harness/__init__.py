"""One bounded Self-Harness iteration; concrete proposal/evaluation are ports.

The application can invoke subsequent iterations using the committed parent.
Memory, skills and tool use are all edits to the same frozen H. No second loop.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from PhysicalRSI_core.contracts import ReconciliationRequired
from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    file_digest,
    locked,
    read_json,
)

from .artifacts import EvolutionSpec, verify_harness
from .selection import validate_profile
from .bindings import policy_identity, validate_admission, validate_round_context
from .population import PopulationRecord
from .skill_mutation import SkillMutation
from .skill_binding import validate_candidate_skills
from .compatibility import ENGINE_VERSION, same_execution


def now():
    return datetime.now(timezone.utc).isoformat()


class Proposer(Protocol):
    def identity(self) -> dict: ...
    def develop(self, parent: dict, output: Path) -> dict: ...
    def propose(self, parent: dict, feedback: dict, output: Path) -> list[dict]: ...


class Evaluator(Protocol):
    def identity(self) -> dict: ...
    def admit(self, candidate: dict, output: Path) -> dict: ...
    def validation(self, comparison: dict, output: Path) -> dict: ...
    def evaluate(
        self, candidate: dict, comparison: dict, cohort: dict, output: Path
    ) -> dict: ...


class SelfHarness:
    def __init__(
        self,
        root,
        *,
        state,
        proposer: Proposer,
        evaluator: Evaluator,
        selector,
        profile,
        protocol,
        scope,
        max_candidates=3,
        round_context=None,
        evolution: EvolutionSpec | None = None,
        archive=None,
    ):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise ValueError("Bounded candidate budget required")
        validate_profile(profile)
        self.evolution = evolution if evolution is not None else EvolutionSpec()
        if not isinstance(self.evolution, EvolutionSpec):
            raise ValueError("Declare an EvolutionSpec")
        if round_context is not None:
            validate_round_context(round_context)
            if not callable(getattr(proposer, "propose_with_context", None)) or not callable(getattr(evaluator, "register_round", None)):
                raise ValueError("Research rounds require context-aware proposal and split registration ports")
        self.root = Path(root).resolve()
        self.archive = archive
        if archive is not None:
            from .archive import CandidateArchive
            if (not isinstance(archive, CandidateArchive) or archive.root.is_relative_to(self.root)
                    or self.root.is_relative_to(archive.root)):
                raise ValueError("Declare a candidate archive outside round evidence")
            proposal_archive = getattr(proposer, "archive", None)
            if proposal_archive is not None and proposal_archive.identity() != archive.identity():
                raise ValueError("Proposal and round must share their archive")
        self.state, self.proposer, self.evaluator, self.selector = (
            state,
            proposer,
            evaluator,
            selector,
        )
        self.population_record = None
        self.config = dict(
            engine_version=ENGINE_VERSION,
            profile=deepcopy(profile),
            protocol=deepcopy(protocol),
            scope=scope,
            max_candidates=max_candidates,
            identities=self.identities(),
            evolution=self.evolution.identity(),
        )
        self.round_context = deepcopy(round_context)
        self.policy = deepcopy(self.config)
        if round_context is not None:
            self.policy = self._campaign_policy()
            self.config.update(schema="physicalrsi.self-harness/v2",
                               campaign_policy=self.policy, round_context=self.round_context)
        self._runtime_identities = deepcopy(self.identities())
        self._runtime_policy = deepcopy(self._campaign_policy()) if round_context is not None else None
        with locked(self.root / ".init.lock"):
            path = self.root / "config.json"
            if path.exists():
                saved = read_json(path)
                if not same_execution(saved, self.config):
                    raise ValueError("Frozen experiment configuration changed")
                self.config = saved
                self.policy = deepcopy(saved.get("campaign_policy", saved))
            else:
                if (self.state.root / "current.json").exists():
                    saved_policy = self.state.resolve()["policy"]
                    if not same_execution(saved_policy, self.policy):
                        raise ValueError("Frozen campaign policy changed")
                    self.policy = deepcopy(saved_policy)
                    if round_context is None:
                        self.config = deepcopy(saved_policy)
                    else:
                        self.config["campaign_policy"] = deepcopy(saved_policy)
                atomic_json(path, self.config)

    def identities(self):
        result = dict(
            proposer=self.proposer.identity(),
            evaluator=self.evaluator.identity(),
            selector=self.selector.identity(),
            core=file_digest(Path(__file__)),
            artifacts=file_digest(Path(__file__).with_name("artifacts.py")),
            scoring=file_digest(Path(__file__).with_name("selection.py")),
        )
        if self.archive is not None:
            result["archive"] = self.archive.identity()
        return result

    def _finish(self, current):
        if self.archive is not None:
            from .evaluation import freeze_json
            archived = self.archive.capture_round(self.root, state=self.state, selector=self.selector)
            if archived["result_revision"] != current["revision"]:
                raise ValueError("Archived round differs from the selected state")
            self._stable()
            path = self.root / "archive.json"
            if path.exists():
                from .archive import verify_archive_receipt
                verify_archive_receipt(self.root)
                if read_json(path)["revision"] != archived["revision"]:
                    raise ValueError("Archive receipt differs from the completed round")
            else:
                freeze_json(path, dict(archive=self.archive.identity(), revision=archived["revision"]))
        return current

    def _campaign_policy(self):
        protocol = {key: value for key, value in self.config["protocol"].items() if key != "identity"}
        identities = self.identities()
        identities.update(proposer=policy_identity(self.proposer), evaluator=policy_identity(self.evaluator),
                          selector=policy_identity(self.selector),
                          bindings=file_digest(Path(__file__).with_name("bindings.py")),
                          lineage=file_digest(Path(__file__).parent.parent / "lineage" / "__init__.py"))
        return dict(schema="physicalrsi.campaign-policy/v2", engine_version=ENGINE_VERSION,
                    profile=deepcopy(self.config["profile"]),
                    protocol=deepcopy(protocol), scope=self.config["scope"],
                    max_candidates=self.config["max_candidates"], identities=identities,
                    evolution=self.evolution.identity())

    def _stable(self):
        if digest(read_json(self.root / "config.json")) != digest(self.config):
            raise ValueError("Frozen experiment configuration changed")
        if digest(self.evolution.identity()) != digest(self.config["evolution"]):
            raise ValueError("Frozen evolution specification changed")
        if digest(self.identities()) != digest(self._runtime_identities):
            raise ValueError("Executable port or core identity changed")
        if self.round_context is not None and digest(self._campaign_policy()) != digest(self._runtime_policy):
            raise ValueError("Campaign policy changed")

    def _step(self, name, inputs, call, *, recover=None):
        # Adapted from v1 RSI._step: completed work resumes, unknown work blocks.
        self._stable()
        path = self.root / (name + ".json")
        identity = digest(inputs)
        if path.exists():
            record = read_json(path)
            if record["input_sha256"] != identity:
                raise ValueError("Checkpoint inputs changed: " + name)
            if record["state"] == "completed":
                if digest(record["output"]) != record["output_sha256"]:
                    raise ValueError("Checkpoint output changed")
                return deepcopy(record["output"])
            if record["state"] != "started" or recover is None:
                raise ReconciliationRequired(
                    "Reconcile uncertain stage before retry: " + str(path)
                )
            # Explicit port recovery owns its finer-grained durable receipts.
            # This is never permission to repeat an unknown physical effect.
            call = recover
            record = dict(record, recovery_attempts=record.get("recovery_attempts", 0) + 1,
                          resumed_at=now())
        else:
            record = dict(state="started", input_sha256=identity, started_at=now())
        atomic_json(path, record)
        try:
            output = call()
            self._stable()
        except BaseException as error:
            atomic_json(
                path, dict(record, error=type(error).__name__ + ": " + str(error))
            )
            raise
        atomic_json(
            path,
            dict(
                record,
                state="completed",
                output=output,
                output_sha256=digest(output),
                finished_at=now(),
            ),
        )
        return deepcopy(output)

    def run(self):
        with locked(self.root / ".run.lock"):
            return self._run()

    def _recovery(self, port, name, *args):
        method = getattr(port, name, None)
        return (lambda: method(*deepcopy(args), self.root)) if callable(method) else None

    def _run(self):
        anchor_path = self.root / "parent.json"
        if anchor_path.exists():
            anchor = read_json(anchor_path)
        else:
            anchor = self.state.resolve()
            atomic_json(anchor_path, anchor)
        if anchor["policy"] != self.policy or anchor["scope"] != self.config["scope"]:
            raise ValueError("Committed state and experiment policy differ")
        parent = anchor["harness"]
        if self.round_context is not None:
            if self.round_context["parent_revision"] != anchor["revision"]:
                raise ValueError("Round context belongs to another parent")
            self.evaluator.register_round(self.state, digest(self.config), context=deepcopy(self.round_context))
        parent_sha = verify_harness(parent)
        if parent_sha != anchor["freeze_sha256"]:
            raise ValueError("Parent changed")
        done = self.root / "commit.json"
        if done.exists():
            value = read_json(done)
            if value != self.state.resolve():
                raise ValueError(
                    "Interaction completed; current state has since changed"
                )
            population_path = self.root / "population.json"
            if population_path.exists():
                self.population_record = PopulationRecord.read(population_path)
            return self._finish(value)
        archive_view = None
        if self.archive is not None:
            from .evaluation import freeze_json
            view_path = self.root / "archive-view.json"
            archive_view = (read_json(view_path) if view_path.exists() else
                            self.archive.view(scope=self.config["scope"], policy_sha256=digest(self.policy)))
            if (archive_view["scope"] != self.config["scope"]
                    or archive_view["policy_sha256"] != digest(self.policy)):
                raise ValueError("Archive view differs from this round's search policy")
            self.archive.summaries(archive_view)
            freeze_json(view_path, archive_view)
        feedback = self._step(
            "develop",
            parent_sha,
            lambda: self.proposer.develop(deepcopy(parent), self.root),
            recover=self._recovery(self.proposer, "resume_development", parent),
        )
        if feedback.get("split") != "evolve" or not feedback.get("evidence"):
            raise ValueError("Development evidence, including failures, required")
        proposal_inputs = dict(parent=parent_sha, feedback=feedback)
        if archive_view is not None:
            proposal_inputs["archive_view"] = archive_view
        if self.round_context is not None:
            proposal_inputs["research_context"] = self.round_context
        children = self._step(
            "propose",
            proposal_inputs,
            (lambda: self.proposer.propose_with_context(
                deepcopy(parent), deepcopy(feedback), deepcopy(self.round_context), self.root
            )) if self.round_context is not None else (lambda: self.proposer.propose(
                deepcopy(parent), deepcopy(feedback), self.root)),
            recover=(self._recovery(self.proposer, "resume_proposal_with_context", parent, feedback,
                                    self.round_context) if self.round_context is not None else
                     self._recovery(self.proposer, "resume_proposal", parent, feedback)),
        )
        if (
            not isinstance(children, list)
            or not 0 <= len(children) <= self.config["max_candidates"]
        ):
            raise ValueError("Candidate pool is malformed or exceeds declared budget")
        population_path = self.root / "population.json"
        if population_path.exists() or "engine_version" in self.config:
            generation = self.round_context["index"] if self.round_context is not None else 0
            population = PopulationRecord.from_candidates(
                round_id=self.root.name, generation=generation, scope=self.config["scope"],
                parent=parent, children=children,
            )
            if population_path.exists():
                saved_population = PopulationRecord.read(population_path)
                population = replace(population, implementation=saved_population.implementation)
            population.write(population_path)
            self.population_record = population
        if not children:
            if verify_harness(parent) != parent_sha:
                raise ValueError("Parent changed during development")
            current = self.state.resolve()
            if current["revision"] != anchor["revision"]:
                raise ValueError("Committed parent changed during development")
            atomic_json(
                self.root / "retained.json",
                dict(reason="no_changed_candidate", qualification=None),
            )
            return self._finish(current)
        pool = [parent, *children]
        hashes = {h["id"]: verify_harness(h) for h in pool}
        if len(hashes) != len(pool) or len(set(hashes.values())) != len(pool):
            raise ValueError("Duplicate candidate ID or executable content")
        if hashes[parent["id"]] != parent_sha:
            raise ValueError("Parent edited during proposal")
        for child in children:
            validate_candidate_skills(child)
            if "archive_origin" in child:
                if self.archive is None:
                    raise ValueError("Archived proposals require a declared archive")
                self.archive.verify_origin(archive_view, child["archive_origin"])
            if self.round_context is not None:
                used = child.get("used_lessons")
                if (not isinstance(used, list) or any(not isinstance(x, str) for x in used)
                        or len(set(used)) != len(used)
                        or not set(used) <= set(self.round_context["lessons"])
                        or any(not child.get(key) for key in ("hypothesis", "expected_effect", "required_checks"))):
                    raise ValueError("Research proposal needs scoped lessons, hypothesis, effect and checks")
                if (any(not isinstance(child[key], str) or not child[key].strip()
                        for key in ("hypothesis", "expected_effect"))
                        or not isinstance(child["required_checks"], list)
                        or any(not isinstance(check, str) or not check for check in child["required_checks"])):
                    raise ValueError("Research proposal hypotheses and check names must be explicit")
            if child["components"]["foundation"] != parent["components"]["foundation"]:
                raise ValueError("Foundation F must stay fixed")
            self.evolution.compare(parent, child)
            if child.get("parent_sha256") != parent_sha or not all(
                child.get(k)
                for k in ("method", "changes", "environment", "evidence", "costs")
            ):
                raise ValueError("Candidate provenance or costs missing")
            if Path(child["root"]).resolve() == Path(parent["root"]).resolve():
                raise ValueError("Materialize children separately")
        accepted, rejected, screening = [], {}, None
        for h in pool:
            receipt = self._step(
                "admit_" + h["id"],
                hashes[h["id"]],
                lambda h=h: self.evaluator.admit(deepcopy(h), self.root),
            )
            validate_admission(receipt, hashes[h["id"]])
            if receipt["accepted"]:
                accepted.append(h)
            else:
                if h["id"] == parent["id"]:
                    raise ValueError("Incumbent failed admission")
                if not receipt.get("reason"):
                    raise ValueError("Keep rejection reasons")
                rejected[h["id"]] = receipt
        if len(accepted) > 1 and callable(getattr(self.evaluator, "screen_pool", None)):
            screening = self._step(
                "screening", dict(parent=parent_sha, candidates={h["id"]: hashes[h["id"]] for h in accepted}),
                lambda: self.evaluator.screen_pool(deepcopy(parent), deepcopy(accepted), self.root),
                recover=self._recovery(self.evaluator, "resume_screening", parent, accepted),
            )
            if (screening.get("role") != "screening_only" or not screening.get("evidence")
                    or set(screening["candidates"]) != {h["id"] for h in accepted}):
                raise ValueError("Screening must cover the entire admitted pool")
            from PhysicalRSI_core.infra.storage import relative_path
            for name, sha in screening["evidence"].items():
                if relative_path(name).parts[:2] != ("experiments", "screening") or file_digest(self.root / name) != sha:
                    raise ValueError("Screening evidence changed or crossed its stage")
            survivors = []
            for h in accepted:
                row = dict(screening["candidates"][h["id"]], evidence=screening["evidence"])
                validate_admission(row, hashes[h["id"]])
                if row["accepted"]:
                    survivors.append(h)
                elif h["id"] == parent["id"]:
                    raise ValueError("Screening cannot remove the incumbent")
                else:
                    rejected[h["id"]] = row
            accepted = survivors
        self._step(
            "admission",
            dict(candidates=hashes, screening=screening) if screening is not None else hashes,
            lambda: dict(rejected=rejected, accepted=[h["id"] for h in accepted]),
        )
        # Even all-rejected pools evaluate the incumbent on fresh inputs only if
        # there is a comparison. Otherwise retain it without a new qualification.
        if len(accepted) == 1:
            if {h["id"]: verify_harness(h) for h in pool} != hashes:
                raise ValueError("Candidate changed during admission")
            atomic_json(
                self.root / "retained.json",
                dict(reason="all_candidates_rejected", qualification=None),
            )
            current = self.state.resolve()
            if current["revision"] != anchor["revision"]:
                raise ValueError("Committed parent changed during admission")
            return self._finish(current)
        comparison = self._step(
            "freeze",
            dict(pool=hashes, config=self.config),
            lambda: dict(
                schema_version=1,
                round_id=self.root.name,
                parent_id=parent["id"],
                frozen_at=now(),
                candidates={h["id"]: hashes[h["id"]] for h in accepted},
                profile=self.config["profile"],
                protocol_sha256=digest(self.config["protocol"]),
                scope=self.config["scope"],
                evaluation_kind=self.config["protocol"]["evaluation_kind"],
                **(dict(round_config_sha256=digest(self.config),
                        round_index=self.round_context["index"]) if self.round_context is not None else {}),
            ),
        )
        cohort = self._step(
            "validation",
            comparison,
            lambda: self.evaluator.validation(deepcopy(comparison), self.root),
            recover=self._recovery(self.evaluator, "resume_validation", comparison),
        )
        if (
            cohort.get("split") != "validation"
            or not cohort.get("admission_evidence")
            or cohort.get("comparison_sha256") != digest(comparison)
        ):
            raise ValueError("Independent validation must bind the frozen pool")
        generated = datetime.fromisoformat(cohort["generated_at"])
        if (
            generated.tzinfo is None
            or generated > datetime.now(timezone.utc)
            or generated <= datetime.fromisoformat(comparison["frozen_at"])
        ):
            raise ValueError("Validation must follow full pool freeze")
        tasks = self.config["profile"]["tasks"]
        if set(cohort["layouts"]) != set(tasks) or any(
            len(cohort["layouts"][t]) != tasks[t]["episodes"] for t in tasks
        ):
            raise ValueError("Validation coverage mismatch")
        layouts = [s for v in cohort["layouts"].values() for s in v]
        if len(set(layouts)) != len(layouts):
            raise ValueError("Duplicate validation inputs")
        self.state.reserve_validation(digest(comparison), layouts)
        pooled = None
        if callable(getattr(self.evaluator, "evaluate_pool", None)):
            if {p["id"]: verify_harness(p) for p in pool} != hashes:
                raise ValueError("Executable changed before pooled evaluation")
            pooled = self._step(
                "evaluation_pool", dict(comparison=comparison, cohort=cohort),
                lambda: self.evaluator.evaluate_pool(deepcopy(accepted), deepcopy(comparison),
                                                     deepcopy(cohort), self.root),
                recover=self._recovery(self.evaluator, "resume_evaluation_pool", accepted, comparison, cohort),
            )
            if (not isinstance(pooled, list) or len(pooled) != len(accepted)
                    or {r["candidate_id"] for r in pooled} != {h["id"] for h in accepted}):
                raise ValueError("Pooled evaluation must cover every admitted candidate exactly once")
            pooled = {r["candidate_id"]: r for r in pooled}
        results = []
        for h in accepted:
            if {p["id"]: verify_harness(p) for p in pool} != hashes:
                raise ValueError("Executable changed before evaluation")
            result = self._step(
                "evaluate_" + h["id"],
                dict(comparison=comparison, cohort=cohort, candidate=hashes[h["id"]]),
                lambda h=h: deepcopy(pooled[h["id"]]) if pooled is not None else self.evaluator.evaluate(
                    deepcopy(h), deepcopy(comparison), deepcopy(cohort), self.root),
                recover=self._recovery(self.evaluator, "resume_evaluation", h, comparison, cohort),
            )
            if {p["id"]: verify_harness(p) for p in pool} != hashes:
                raise ValueError("Executable changed during evaluation")
            if result.get("evaluator_revision") != self.config["protocol"]["identity"]:
                raise ValueError("Evaluator protocol changed")
            results.append(result)
        if {h["id"]: verify_harness(h) for h in pool} != hashes:
            raise ValueError("Executable changed during comparison")
        # Recompute selection on resume, rechecking every raw evidence digest.
        decision = self.selector.select(
            comparison, cohort, results, evidence_root=self.root
        )
        saved = self._step(
            "selection",
            dict(comparison=comparison, cohort=cohort, results=results),
            lambda: decision,
        )
        if saved != decision:
            raise ValueError("Selection no longer agrees with evidence")
        survivor = next(h for h in accepted if h["id"] == decision["survivor_id"])
        evidence = {
            str(p): file_digest(p)
            for p in self.root.rglob("*.json")
            if p.name != "commit.json"
        }
        # Trajectories contain binary observations/proposals as well as JSON.
        # Preserve the evaluator's complete evidence closure through inheritance.
        from PhysicalRSI_core.infra.storage import relative_path

        for result in results:
            for episode in result["episodes"]:
                for name, sha in episode["evidence_sha256"].items():
                    path = (self.root / relative_path(name)).resolve()
                    if not path.is_relative_to(self.root) or file_digest(path) != sha:
                        raise ValueError("Evaluation evidence changed before commit")
                    evidence[str(path)] = sha
        if screening is not None:
            for name, sha in screening["evidence"].items():
                path = self.root / relative_path(name)
                if file_digest(path) != sha:
                    raise ValueError("Screening evidence changed before commit")
                evidence[str(path)] = sha
        committed = self.state.commit(
            anchor["revision"],
            survivor,
            evidence=evidence,
            decision=decision,
            selector=self.selector,
            comparison_root=self.root,
        )
        atomic_json(done, committed)
        return self._finish(committed)
