"""Retrieve completed comparisons for System 2, including rejected candidates.

Historical validation becomes development knowledge. Its cases must never be
reused by the new campaign. ResearchMemory records observations and hypotheses;
it neither promotes candidates nor changes the native-success selector.
"""
import json
from pathlib import Path

from PhysicalRSI_core.infra.storage import digest, relative_path
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.research_memory import ResearchMemory
from PhysicalRSI_core.self_harness.selection import select_survivor

from .common import file_hash, save
from .candidate_programs import read_programs
from .task_diagnostics import diagnose_trajectory


def read(path):
    return json.loads(Path(path).read_text())


def completed(directory, name):
    path = Path(directory) / (name + ".json")
    value = read(path)
    if value.get("state") != "completed" or digest(value["output"]) != value["output_sha256"]:
        raise ValueError("Historical stage is incomplete or changed: " + str(path))
    return value["output"]


def inspect_round(directory):
    directory = Path(directory).resolve()
    if not (directory / "commit.json").is_file():
        raise ValueError("History requires a completed committed comparison")
    comparison = completed(directory, "freeze")
    cohort = completed(directory, "validation")
    selected = completed(directory, "selection")
    candidates = [read(directory / "parent.json")["harness"], *completed(directory, "propose")]
    admitted = {c["id"]: c for c in candidates if c["id"] in comparison["candidates"]}
    if set(admitted) != set(comparison["candidates"]):
        raise ValueError("Historical candidate pool differs")
    for key, candidate in admitted.items():
        if verify_harness(candidate) != comparison["candidates"][key]:
            raise ValueError("Historical candidate artifacts changed")
    results = [completed(directory, "evaluate_" + key) for key in admitted]
    if select_survivor(comparison, cohort, results, evidence_root=directory) != selected:
        raise ValueError("Historical selection no longer agrees with raw evidence")
    commit = read(directory / "commit.json")
    if (commit["freeze_sha256"] != selected["survivor_freeze_sha256"] or
            verify_harness(commit["harness"]) != commit["freeze_sha256"]):
        raise ValueError("Historical commit does not match selected artifacts")
    names = ["parent", "propose", "freeze", "validation", "selection", "commit"]
    names += ["evaluate_" + key for key in admitted]
    evidence = {str(directory / (name + ".json")): file_hash(directory / (name + ".json")) for name in names}
    trials = []
    seeds = set()
    for path in sorted((directory / "experiments").glob("*/*/trials/*/experiment.json")):
        specification = read(path)
        case = specification["case"]
        if case.get("split") not in {"development", "validation"} or type(case.get("seed")) is not int or case["seed"] <= 2:
            raise ValueError("Final/test cases cannot enter development research history")
        seeds.add(case["seed"])
        trials.append(path)
    if not trials:
        raise ValueError("Historical comparison has no native experiment cases")
    return dict(directory=str(directory), candidates=admitted, results=results,
                selection=selected, evidence=evidence, consumed_seeds=sorted(seeds))


def freeze_imports(root, directories, future_seeds):
    """Pin completed external rounds before new candidates or trials are run."""
    path = Path(root) / "history_imports.json"
    entries = []
    for name in directories:
        item = inspect_round(name)
        if set(item["consumed_seeds"]) & set(future_seeds):
            raise ValueError("Historical development knowledge overlaps future cases")
        entries.append({k: item[k] for k in ("directory", "evidence", "consumed_seeds")})
    if len({e["directory"] for e in entries}) != len(entries):
        raise ValueError("Duplicate history import")
    value = dict(schema="dexjoco.history-imports/v1", rounds=entries,
                 role="Previously completed non-final validation is now development evidence")
    if path.exists():
        if read(path) != value:
            raise ValueError("Frozen history imports changed")
    else:
        save(path, value)
    return value


def _directories(root, output):
    imports = Path(root) / "history_imports.json"
    entries = read(imports)["rounds"] if imports.exists() else []
    paths = []
    for item in entries:
        for name, sha in item["evidence"].items():
            if file_hash(name) != sha:
                raise ValueError("Frozen historical comparison changed")
        paths.append(Path(item["directory"]))
    ledger = Path(root) / "campaign.json"
    if ledger.exists():
        current = Path(output).name
        for record in read(ledger)["rounds"]:
            directory = Path(root) / "rounds" / f"round-{record['index']+1:04d}"
            if directory.name >= current:
                raise ValueError("Future or current round cannot enter its own reflection history")
            for name in ("selection.json", "commit.json"):
                if file_hash(directory / name) != record["files"].get(name):
                    raise ValueError("Completed campaign history changed")
            paths.append(directory)
    return list(dict.fromkeys(p.resolve() for p in paths))


def build_research_feedback(root, output, *, source, tasks, maximum_rounds=4, maximum_lessons=8):
    output = Path(output)
    directories = _directories(root, output)
    omitted = [dict(round=str(p), reason="older than bounded history window") for p in directories[:-maximum_rounds]]
    lessons = {}
    for directory in directories[-maximum_rounds:]:
        old = inspect_round(directory)
        token = digest(old["evidence"])[:16]
        for result in old["results"]:
            candidate = old["candidates"][result["candidate_id"]]
            candidate_root = Path(candidate["root"])
            memory = (candidate_root / "memory.md").read_text()
            programs = read_programs(candidate_root) if (candidate_root / "programs.json").exists() else []
            for episode in result["episodes"]:
                if episode["task"] not in tasks:
                    continue
                names = [n for n in episode["evidence_sha256"] if n.endswith("/trajectory.json")]
                if len(names) != 1:
                    raise ValueError("History needs one verified trajectory per episode")
                trajectory = directory / relative_path(names[0])
                diagnostic_path = output / "history_diagnostics" / token / candidate["id"] / (trajectory.parent.name + ".json")
                diagnosis = diagnose_trajectory(trajectory, source, diagnostic_path)
                key = "history-" + digest(dict(round=token, candidate=candidate["id"], task=episode["task"]))[:24]
                evidence = dict(old["evidence"])
                evidence.update(diagnosis["binding"]["inputs"])
                evidence[str(diagnostic_path.resolve())] = file_hash(diagnostic_path)
                evidence[str(candidate_root / "memory.md")] = file_hash(candidate_root / "memory.md")
                for name, sha in candidate["components"]["control"].items():
                    if name.startswith("programs/") or name == "programs.json":
                        evidence[str(candidate_root / name)] = sha
                selected = old["selection"]["survivor_id"] == candidate["id"]
                lessons[key] = dict(
                    conditions=dict(task=episode["task"], environment="dexjoco", track="astra-direct"),
                    hypothesis="Use this completed attempt to revise the next hypothesis. Rejection is evidence about this tested candidate and case, not proof that every component is ineffective.",
                    required_checks=["fresh-paired-native-success"],
                    counterexample_checks=["recheck-observed-bottlenecks"],
                    evidence=evidence,
                    observation=dict(candidate_id=candidate["id"], was_selected=selected,
                        native_success=episode["success"], score=episode["score"],
                        selection_decision=old["selection"]["decision"],
                        rejection=old["selection"].get("rejected", {}).get(candidate["id"]),
                        memory_excerpt=memory[:6000], memory_truncated=len(memory)>6000,
                        programs=[dict(name=p["name"], usage=p["usage"], source=p["source"][:6000],
                                       source_truncated=len(p["source"])>6000) for p in programs],
                        diagnostic=diagnosis["summary"], prior_round=str(directory)))
    if len(lessons) > maximum_lessons:
        discarded = list(lessons)[:-maximum_lessons]
        omitted.extend(dict(lesson=k, reason="bounded lesson count") for k in discarded)
        lessons = {k: v for k, v in lessons.items() if k not in discarded}
    result = dict(role="System 2 research history; failed candidates remain evidence, never automatically accepted policy memory",
                  omissions=omitted, lessons={}, revision=None, snapshot_path=None)
    if lessons:
        store = ResearchMemory(output / "research_memory")
        revision = store.write(lessons)
        retrieved = {}
        for task in tasks:
            retrieved.update(store.retrieve(revision, dict(task=task, environment="dexjoco", track="astra-direct")))
        result.update(lessons=retrieved, revision=revision,
                      snapshot_path=str((store.root / (revision + ".json")).resolve()))
    save(output / "research_history.json", result)
    return result
