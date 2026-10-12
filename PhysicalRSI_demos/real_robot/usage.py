"""Separate controller entries, model calls, actions and human assistance."""


def usage(events):
    entries = {level: 0 for level in ("L1", "L2", "L3")}
    models, actions = {}, dict(entries)
    interventions = dict(language=0, video_reference=0, teleoperation=0)
    consultation_ids = {e.get("intervention_id") for e in events
                        if e["kind"] == "video_consultation_started" and isinstance(e.get("intervention_id"), str)}
    counted_consultations = set()
    attempts = closures = 0
    invalid = 0
    for event in events:
        kind, level = event["kind"], event.get("level")
        if kind in {"controller_enter", "action_started"} and (not isinstance(level, str) or level not in entries):
            invalid += 1
            continue
        if kind == "controller_enter":
            entries[level] += 1
        elif kind == "model_request":
            role = event.get("role")
            if not isinstance(role, str) or role not in {"monitor", "execution", "development", "perception"}:
                invalid += 1
                continue
            models[role] = models.get(role, 0) + 1
        elif kind == "action_started":
            actions[level] += 1
            if event.get("method") == "gripper" and event.get("arguments", {}).get("action") == "close":
                closures += 1
        elif kind == "attempt_started":
            attempts += 1
        elif kind == "human_guidance":
            interventions["language"] += 1
        elif kind == "video_consultation_started" and event.get("intervention_id") not in counted_consultations:
            counted_consultations.add(event.get("intervention_id"))
            interventions["video_reference"] += 1
        elif kind == "video_reference" and event.get("intervention_id") not in consultation_ids:
            interventions["video_reference"] += 1
        elif kind == "human_handoff":
            interventions["teleoperation"] += 1
    return dict(schema="tianji.controller-usage/v1", controller_entries=entries,
                model_requests=models, actions_started=actions, attempts=attempts,
                closure_attempts=closures, l3_interventions=interventions,
                l3_uses=sum(interventions.values()), invalid_usage_events=invalid,
                semantics="Controller entries, model requests and interventions are distinct counts")
