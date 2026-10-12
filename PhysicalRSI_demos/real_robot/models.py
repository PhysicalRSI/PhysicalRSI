"""Separate model sessions with explicit payloads; no model gets host file tools."""
import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import threading
import uuid

from .contracts import Frame
from .evidence import write_new
from .references import use_physicalrsi


def image_messages(model, instruction, payload, frames=()):
    responses = model.config.protocol == "responses"
    content = [dict(type="input_text" if responses else "text",
                    text=json.dumps(payload, ensure_ascii=False, allow_nan=False))]
    for frame in frames:
        frame.validate()
        data = Path(frame.path).read_bytes()
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            raise ValueError("Camera attachment must be PNG or JPEG")
        url = "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
        content.append(dict(type="input_image", image_url=url) if responses
                       else dict(type="image_url", image_url=dict(url=url)))
    return [dict(role="system", content=instruction), dict(role="user", content=content)]


def recent_image_history(history, *, keep=2):
    """Retain recent image groups without dropping text or tool-call pairs.

    The episode owns immutable full RGB evidence. Older attachments are omitted
    from the next model request only; their observation IDs and results remain.
    Supports both upstream chat-completions and Responses content blocks.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError("At least one current image group is required")
    image_types = {"image_url", "input_image"}
    groups = [i for i, item in enumerate(history)
              if isinstance(item.get("content"), list) and any(
                  block.get("type") in image_types for block in item["content"])]
    omitted = set(groups[:-keep])
    result = []
    for i, item in enumerate(history):
        if i not in omitted:
            result.append(item)
            continue
        content = [block for block in item["content"]
                   if block.get("type") not in image_types]
        responses = any(block.get("type") == "input_image" for block in item["content"])
        content.append(dict(type="input_text" if responses else "text",
                            text="Earlier RGB attachments omitted from this request; "
                                 "full images remain in episode evidence. Use the latest observation for actions."))
        result.append(dict(item, content=content))
    return result


def json_answer(model, instruction, payload, frames=()):
    text, calls, _ = model.complete(image_messages(model, instruction, payload, frames), [])
    if calls:
        raise ValueError("This model role has no tools")
    return json.loads(text)


@dataclass(frozen=True)
class MailboxConfig:
    model: str
    session: str
    protocol: str = "chat_completions"
    timeout_s: float = 600

    def public(self):
        return dict(kind="external-model-mailbox", **asdict(self))


class MailboxModel:
    """Explicit external GPT worker interface, also usable by the current agent.

    Requests are not answered by templates. A response must identify the model
    session and request. No retries or stale-response adoption.
    """
    def __init__(self, root, config: MailboxConfig, *, role):
        self.root = Path(root).resolve() / role
        self.root.mkdir(parents=True, exist_ok=True)
        self.config, self.role = config, role
        self.cancelled = threading.Event()
        if not config.model or not config.session or config.timeout_s <= 0:
            raise ValueError("Mailbox model identity/session/timeout required")

    def complete(self, history, tools):
        request_id = uuid.uuid4().hex
        folder = self.root / request_id
        request = dict(id=request_id, role=self.role, model=self.config.public(),
                       history=history, tools=tools)
        write_new(folder / "request.json", request)
        print(f"[{self.role}] GPT 请求：{folder / 'request.json'}", flush=True)
        deadline = time.monotonic() + self.config.timeout_s
        response = folder / "response.json"
        while not response.exists():
            if self.cancelled.is_set():
                raise RuntimeError("Model request cancelled after task stop")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPT {self.role} response timed out")
            time.sleep(0.05)
        value = json.loads(response.read_text())
        if (set(value) != {"id", "model", "session", "text", "calls"}
                or value["id"] != request_id or value["model"] != self.config.model
                or value["session"] != self.config.session or not isinstance(value["text"], str)
                or not isinstance(value["calls"], list)):
            raise ValueError("Model response identity or schema differs")
        native = dict(role="assistant", content=value["text"])
        if value["calls"]:
            native["tool_calls"] = [dict(id=c["id"], type="function", function=dict(
                name=c["name"], arguments=c["arguments"])) for c in value["calls"]]
        return value["text"], value["calls"], [native]

    def close(self):
        self.cancelled.set()

    def tool_result(self, call_id, value):
        return dict(role="tool", tool_call_id=call_id, content=json.dumps(value, ensure_ascii=False))


def make_model(spec, *, role, workspace):
    if spec["kind"] == "mailbox":
        return MailboxModel(Path(workspace) / "model_requests", MailboxConfig(**spec["config"]), role=role)
    if spec["kind"] == "http":
        use_physicalrsi()
        from PhysicalRSI_core.infra.language import LanguageModel, ModelConfig
        return LanguageModel(ModelConfig(**spec["config"]))
    raise ValueError("Model kind must be mailbox or http")


class VisualCodeModel:
    """Attach development sensor images to the original CodeProposer request.

    Upstream CodeProposer sends JSON text; this wrapper supplies the actual RGB
    pixels as well, without giving the proposer file-system access.
    """
    def __init__(self, model, *, output=None, reusable_skills=()):
        self.model, self.config = model, model.config
        self.output = Path(output) if output is not None else None
        self.reusable_skills = list(reusable_skills)

    def complete(self, history, tools):
        request = json.loads(history[-1]["content"])
        request["reusable_skills"] = self.reusable_skills
        frames = []
        seen = set()
        for row in request["development_feedback"]:
            if row["split"] != "development":
                raise ValueError("Validation data cannot enter the proposer")
            for obs in row.get("observations", []):
                for item in obs["frames"]:
                    frame = Frame(**item)
                    if frame.sha256 not in seen:
                        frames.append(frame)
                        seen.add(frame.sha256)
            for event in row.get("execution", []):
                if event["kind"] == "video_reference":
                    for item in event["reference"]["frames"]:
                        frame = Frame(**{k: item[k] for k in ("camera", "path", "captured_ns", "sha256")})
                        if frame.sha256 not in seen:
                            frames.append(frame)
                            seen.add(frame.sha256)
        instruction = history[0]["content"] + (
            " Write a reusable, observation-conditioned task program. Derive targets "
            "from the current scene, object relationships and task parameters; do not "
            "require exact metric object positions when visual relations suffice. robot.look "
            "returns RGB feature estimates for your code to interpret, not a GPT action planner. "
            "robot.align expresses a multi-view image alignment goal; it is translation-only and "
            "returns residual/unknown when local visual control is unsupported. robot.relate supports "
            "multi-landmark pose alignment and object-sized clearance using current feature vectors. "
            "Image alignment is not grasp or contact: reobserve visible grasp geometry before choosing "
            "closure, then reassess the result; never claim a grasp from tool completion alone. "
            "Use robot.gripper(arm, action, mode=None): open/close/hold/release, wrap/tripod modes. "
            "The host coordinates configured hand poses and returns measured execution status. "
            "Never emit six motor targets or copy a raw L3 hand trajectory into task code. "
            "Manufacturer-derived initial profiles allow grasping without demonstration. "
            "Do not request L3 merely because a profile is not yet physically validated. "
            "Only use motion capabilities declared available; look alone cannot move an arm. Do not "
            "replay fixed centimetre offsets or a recorded list of joint targets. "
            "Learn from L2 and L3 demonstrations by extracting preconditions, relative "
            "geometry, reusable skills and recovery rules. A demonstration is evidence, "
            "not a trajectory to copy. If the available API cannot support the task, "
            "call robot.yield_control with a public explanation. Do not encode judge "
            "answers as policy observations. Independent evaluation decides promotion."
            " Historical video_reference images are demonstrations, NEVER the current robot scene. "
            "Their timestamps are video-relative labels. Extract relative relationships and phases, "
            "then reacquire current RGB before any action. No recorded numeric controls are supplied."
            " The reusable_skills field supplies existing executable L1 skills on EVERY proposal, "
            "including when the current parent accidentally omitted them. Reuse or adapt applicable "
            "helper functions and call them from the task program; a prose summary followed by yield "
            "is not implementation reuse. Preserve phase checks: never reopen an already held object. "
            "The source's old task entry is an example, not the new task instruction. Explain any "
            "inapplicability in rationale. A missing later phase such as uncapping does not by itself "
            "invalidate a supported earlier phase; check that phase's current visual preconditions. "
            "The code validator rejects imports, dangerous builtins, and ALL dunder attributes, "
            "including type(error).__name__. Use str(error) for exception diagnostics. "
            "Define exactly policy(robot, memory); no filesystem, network or trajectory replay."
        )
        from .policy_runtime import validate_source
        messages = image_messages(self.model, instruction, request, frames)
        for attempt in range(1, 4):
            result = self.model.complete(messages, tools)
            answer, calls, _ = result
            candidate = None
            try:
                if calls:
                    raise ValueError("Code proposal must be JSON, not a tool call")
                candidate = json.loads(answer)
                if (not isinstance(candidate, dict) or set(candidate) != {"source", "memory", "rationale"}
                        or not isinstance(candidate["source"], str)
                        or not isinstance(candidate["memory"], dict)
                        or not isinstance(candidate["rationale"], str)):
                    raise ValueError("Expected source string, memory object and rationale string")
                validate_source(candidate["source"])
            except (ValueError, SyntaxError) as exc:
                if self.output is not None:
                    write_new(self.output / f"source-validation-{attempt:02d}.json",
                              dict(valid=False, response=answer, calls=calls,
                                   error_type=type(exc).__name__, validation_error=str(exc), executed=False))
                if attempt == 3:
                    raise
                # Only the static code-validation error is returned. No private
                # monitor data, robot state or failed candidate is executed.
                repair = dict(validation_error=str(exc), executed=False,
                    rejected_source=candidate.get("source") if isinstance(candidate, dict) else None,
                    rejected_response=answer,
                    instruction="Repair this rejected proposal. Return the complete corrected source, memory and rationale JSON; retain reusable skill code and task logic.")
                messages = messages + [dict(role="user", content=json.dumps(repair, ensure_ascii=False))]
            else:
                if self.output is not None:
                    write_new(self.output / f"source-validation-{attempt:02d}.json",
                              dict(valid=True, response=answer, executed=False))
                return result


class RecordedModel:
    """Count all roles without copying private monitor text into execution."""
    def __init__(self, model, episode, role):
        self.model, self.episode, self.role = model, episode, role
        self.config = model.config

    def complete(self, history, tools):
        level = "L2" if self.role == "execution" else "L1"
        post = (self.episode.root / "result.json").exists()
        # Finished event chains are immutable. Post-episode learning has its own
        # receipt and cannot invalidate the result's execution hash.
        index = uuid.uuid4().hex if post else self.episode.event("model_request", level=level, role=self.role)
        def record(status, **fields):
            if post:
                write_new(self.episode.root / "post-episode-model-requests" / (index + ".json"),
                          dict(role=self.role, status=status, **fields))
            else:
                self.episode.event("model_result", level=level, role=self.role,
                                   request_event=index, status=status, **fields)
        try:
            result = self.model.complete(history, tools)
        except Exception as exc:
            record("error", error_type=type(exc).__name__)
            raise
        record("completed")
        return result

    def tool_result(self, call_id, value):
        return self.model.tool_result(call_id, value)

    def close(self):
        close = getattr(self.model, "close", None)
        if close:
            close()


def require_independent_models(models):
    """Check the bundled independent HTTP transport for physical execution.

    Custom transports implement Model and must supply their own role-isolation
    admission check at the host boundary; offline mailboxes are never admitted.
    """
    import os
    from PhysicalRSI_core.infra.language import LanguageModel
    for role, model in models.items():
        if not isinstance(model, LanguageModel):
            raise ValueError("Physical execution requires independent HTTP model roles; "
                             "mailboxes or custom transports need host isolation verification")
        if model.config.model.startswith("YOUR_") or not os.environ.get(model.config.api_key_env):
            raise ValueError(f"Configure the {role} model and its API key environment variable before hardware startup")
