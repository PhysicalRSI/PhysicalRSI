"""RGB/semantic learning projection. Raw controls remain audit evidence only."""
from .contracts import json_copy


def proposal_feedback(rows):
    """Construct an allowlist, never pass arbitrary recorded dictionaries through.

    Terminal labels supervise development; private assessment text is absent.
    Both L2 and L3 contribute images, visible intent and outcome, without exposing
    recorded motor targets, world EEF waypoints or measured joint trajectories.
    """
    projected = []
    for row in rows:
        if row.get("split") != "development":
            raise ValueError("Proposal accepts development evidence only")
        value = {k: json_copy(row[k]) for k in
                 ("task", "split", "instruction", "outcome", "solved_by", "l2_calls", "l3_calls", "capabilities")
                 if k in row}
        if "capabilities" not in value:
            declared = [e["capabilities"] for e in row.get("execution", [])
                        if e["kind"] == "execution_capabilities"]
            if declared:
                value["capabilities"] = json_copy(declared[-1])
        observations = row.get("observations", [])
        window = observations[-64:]
        selected = sorted({round(i * (len(window) - 1) / 7) for i in range(8)}) if window else []
        value["observations"] = [{k: json_copy(window[i][k]) for k in ("id", "captured_ns", "frames")}
                                 for i in selected]
        value["evidence_selection"] = dict(total_observations=len(observations),
            selected_observation_ids=[o["id"] for o in value["observations"]],
            policy="eight temporal samples from latest 64 observations; full evidence retained on host")
        value["execution"] = []
        events = row.get("execution", [])
        video_indices = [i for i,e in enumerate(events) if e["kind"] == "video_reference"][-2:]
        for index, event in enumerate(events):
            item = {k: json_copy(event[k]) for k in
                    ("index", "kind", "level", "observation_id", "intent", "method", "reason") if k in event}
            if event.get("kind") == "action_started":
                args = event.get("arguments", {})
                item["visible_purpose"] = args.get("reason", "")
                if event.get("method") == "gripper":
                    item["gripper"] = {k: args[k] for k in ("arm", "action", "mode") if k in args}
            if event.get("kind") == "action_completed":
                receipt = event.get("receipt", {})
                item["execution_status"] = receipt.get("status")
                item["gripper_status"] = receipt.get("gripper_result", {}).get("state")
            if event.get("kind") == "human_guidance":
                item["guidance"] = event["guidance"]
            if event.get("kind") == "video_reference":
                # Historical RGB is labeled separately and never made a current
                # robot observation or submitted to the task monitor.
                if index in video_indices:
                    item["reference"] = json_copy(event["reference"])
                else:
                    item["kind"] = "earlier_video_reference"
                    item["attachment_omitted"] = True
            value["execution"].append(item)
        projected.append(value)
    return projected
