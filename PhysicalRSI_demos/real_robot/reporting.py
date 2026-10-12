"""Read-only episode accounting, including missing or incomplete final receipts.

Explicit episode directories define the cohort. No private evaluation, pixels,
model, ROS or hardware is opened. This is accounting, not policy qualification.
"""
import json
from pathlib import Path

from .contracts import Task, digest
from .evidence import metrics
from .usage import usage


def _object(path):
    if path.is_symlink():
        raise ValueError("symlink_record")
    def reject_constant(_):
        raise ValueError("non_finite_json")
    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("non_object_record")
    return value


def _read_episode(root):
    issues = []
    row = dict(path=str(root), task=None, task_sha256=None, split="unknown",
               qualification="unknown", policy_id=None, outcome="uncertain",
               solved_by=None, l1_success=False, l2_calls=0, l3_calls=0,
               human_judgments=0, receipt="missing", status="unknown",
               counts_complete=False, human_judgments_known=False, issues=issues)
    metadata = None
    try:
        metadata = _object(root / "episode.json")
        task = Task(**metadata["task"]).public()
        if (metadata.get("schema") != "tianji.dual-system.episode/v1"
                or metadata.get("split") not in {"development", "validation"}
                or metadata.get("qualification") not in {"offline", "physical"}
                or metadata.get("policy_id") is not None
                and not isinstance(metadata["policy_id"], str)):
            raise ValueError("invalid_metadata")
        row.update(task=task, task_sha256=digest(task), split=metadata["split"],
                   qualification=metadata["qualification"], policy_id=metadata.get("policy_id"))
    except (OSError, ValueError, KeyError, TypeError):
        metadata = None
        issues.append("metadata_missing_or_invalid")

    events = []
    events_valid = True
    public = root / "execution"
    if public.is_symlink() or not public.is_dir():
        events_valid = False
        issues.append("execution_missing_or_symlink")
    else:
        for index, path in enumerate(sorted(public.glob("event-*.json"))):
            try:
                event = _object(path)
                if (path.name != f"event-{index:06d}.json"
                        or type(event.get("index")) is not int or event["index"] != index
                        or not isinstance(event.get("kind"), str)):
                    raise ValueError("invalid_event")
                if event["kind"] == "controller_enter":
                    if event.get("level") not in {"L1", "L2", "L3"}:
                        raise ValueError("invalid_controller")
                    if event["level"] != "L1":
                        row[event["level"].lower()+"_calls"] += 1
                if event["kind"] == "l1_policy_selected":
                    if not isinstance(event.get("policy_id"), str):
                        raise ValueError("invalid_policy")
                    row["policy_id"] = event["policy_id"]
                events.append(event)
            except (OSError, ValueError, KeyError, TypeError):
                events_valid = False
                issues.append("event_history_incomplete_or_invalid")
                # Only count the verified contiguous prefix, never guess gaps.
                break

    result_path = root / "result.json"
    if not result_path.exists() and not result_path.is_symlink():
        issues.append("result_missing")
        return row
    row["receipt"] = "invalid"
    try:
        result = _object(result_path)
        if (metadata is None or not events_valid
                or result.get("schema") != "tianji.dual-system.result/v1"
                or result.get("task") != row["task"]["id"]
                or any(result.get(k) != row[k] for k in ("split", "qualification", "policy_id"))
                or result.get("outcome") not in {"success", "failure", "uncertain"}
                or result.get("solved_by") not in {None, "L1", "L2", "L3"}
                or type(result.get("l1_success")) is not bool
                or result["l1_success"] != (result["outcome"] == "success" and result["solved_by"] == "L1"
                    and (result.get("attribution") != "unassisted-L1/v2" or
                         not row["l2_calls"] and not row["l3_calls"] and
                         not any(e["kind"] == "human_guidance" for e in events)))
                or result["outcome"] != "success" and result["solved_by"] is not None
                or not isinstance(result.get("status"), str)
                or result.get("events_sha256") != digest(events)):
            raise ValueError("result_mismatch")
        for key in ("l2_calls", "l3_calls", "human_judgments"):
            if type(result.get(key)) is not int or result[key] < 0:
                raise ValueError("invalid_count")
        if any(result[k] != row[k] for k in ("l2_calls", "l3_calls")):
            raise ValueError("controller_count_mismatch")
        for key in ("outcome", "solved_by", "l1_success", "status", "human_judgments"):
            row[key] = result[key]
        row.update(receipt="verified", counts_complete=True, human_judgments_known=True)
    except (OSError, ValueError, KeyError, TypeError):
        issues.append("result_incomplete_or_inconsistent")
    return row


def detailed_usage(root):
    """Append-only historical reconciliation; never rewrite legacy result hashes."""
    events = []
    complete = True
    for index, path in enumerate(sorted((root / "execution").glob("event-*.json"))):
        try:
            event = _object(path)
            if event["index"] != index or path.name != f"event-{index:06d}.json":
                raise ValueError("Noncontiguous events")
            events.append(event)
        except (ValueError, OSError, KeyError, TypeError):
            complete = False
            break
    counts = usage(events)
    adjustments = []
    # These legacy records were saved outside the episode's signed event chain.
    # Report provenance and distinguish reconstructed uses from signed counts.
    if root.name == "episode":
        for path in sorted((root.parent / "operator-interventions").glob("l3-language-*.json")):
            try:
                value = _object(path)
                if (value.get("schema") != "tianji.operator-intervention/v1" or value.get("level") != "L3"
                        or value.get("type") != "human_language_guidance" or value.get("count") != 1):
                    raise ValueError("Unrecognized historical guidance")
                if value.get("run_id", root.parent.name) != root.parent.name:
                    raise ValueError("Guidance belongs to another run")
                import hashlib
                adjustments.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        subtype="language", count=1))
            except (ValueError, OSError, KeyError, TypeError):
                complete = False
    counts["legacy_sidecar_adjustments"] = adjustments
    counts["l3_uses_including_legacy_guidance"] = counts["l3_uses"] + len(adjustments)
    counts["event_prefix_complete"] = complete and not counts["invalid_usage_events"]
    counts["model_requests_recorded"] = any(e["kind"] == "model_request" for e in events)
    # Absence of instrumentation in old files does not establish zero calls.
    if not counts["model_requests_recorded"]:
        counts["model_requests"] = None
    return counts


def report_episodes(directories):
    """One row per distinct existing episode directory; never silently omit it.

    Missing receipts do not establish whether an experiment is running, waiting,
    interrupted, or stopped. Unknown counts are reported as lower bounds.
    """
    rows, duplicates, rejected, seen = [], [], [], set()
    for directory in directories:
        path = Path(directory).resolve()
        if path in seen:
            duplicates.append(str(path))
            continue
        seen.add(path)
        if not path.is_dir() or not any((path / n).exists() or (path / n).is_symlink()
                                       for n in ("episode.json", "execution", "result.json")):
            rejected.append(dict(path=str(path), reason="not_an_episode_directory"))
            continue
        row = _read_episode(path)
        row["usage"] = detailed_usage(path)
        rows.append(row)
    groups = {}
    for row in rows:
        key = (row["task_sha256"], row["qualification"], row["split"])
        groups.setdefault(key, []).append(row)

    def summarize(cohort):
        result = metrics(cohort)
        result.update(denominator="all distinct supplied episode directories in this group",
                      receipt_missing=sum(r["receipt"] == "missing" for r in cohort),
                      receipt_invalid=sum(r["receipt"] == "invalid" for r in cohort),
                      call_counts_are_lower_bounds=any(not r["counts_complete"] for r in cohort),
                      human_judgments_unknown_episodes=sum(not r["human_judgments_known"] for r in cohort))
        result["l3_uses_including_legacy_guidance"] = sum(r["usage"]["l3_uses_including_legacy_guidance"] for r in cohort)
        result["controller_entry_semantics"] = "l2_calls/l3_calls retain legacy entry counts; usage separates model/action/human counts"
        return result

    summaries = []
    for cohort in groups.values():
        policies = {}
        for row in cohort:
            policies.setdefault(row["policy_id"], []).append(row)
        summaries.append(dict(task=cohort[0]["task"], task_sha256=cohort[0]["task_sha256"],
                              qualification=cohort[0]["qualification"], split=cohort[0]["split"],
                              metrics=summarize(cohort),
                              policies=[dict(policy_id=pid, metrics=summarize(records))
                                        for pid, records in policies.items()]))
    return dict(schema="tianji.episode-report/v1", episodes=len(rows), groups=summaries,
                records=rows, duplicate_inputs=duplicates, rejected_inputs=rejected,
                scope="Explicit input cohort only; no process status or physical qualification inferred")
