"""Data-only language-model proposals for a declared numeric search space.

The model is asked for JSON edits to fields declared by :class:`ParameterSpace`.
The response is treated as untrusted data: every value is checked, quantized and
materialized through the normal harness publication path before it can become a
candidate.
"""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from typing import Protocol

from ..infra.language import LanguageModel, ModelConfig
from ..infra.storage import atomic_json, digest, file_digest, read_json
from ..trials import task_case_identity
from .artifacts import verify_harness
from .search_space import ParameterSpace


class ModelClient(Protocol):
    def identity(self) -> dict: ...

    def complete(self, messages: list[dict], *, max_output_tokens: int) -> dict: ...


class ScriptedModelClient:
    """A deterministic model boundary for tests and offline replay."""

    def __init__(self, responses: list[str]):
        if not isinstance(responses, list) or any(not isinstance(value, str) for value in responses):
            raise ValueError("Scripted model responses must be a list of strings")
        self.responses = list(responses)
        self._index = 0

    def identity(self):
        return dict(kind="scripted-model-client/v1", scripts_sha256=digest(self.responses),
                    count=len(self.responses))

    def complete(self, messages: list[dict], *, max_output_tokens: int):
        if self._index >= len(self.responses):
            raise RuntimeError("Scripted model has no response remaining")
        text = self.responses[self._index]
        self._index += 1
        return dict(text=text, usage=dict(input_tokens=0, output_tokens=0))


class HTTPModelClient:
    """Adapter for the small HTTP language boundary in ``infra.language``."""

    def __init__(self, config: ModelConfig):
        if not isinstance(config, ModelConfig):
            raise ValueError("HTTP model client requires a ModelConfig")
        self.config = config
        self.model = LanguageModel(config)

    def identity(self):
        return self.config.public()

    def complete(self, messages: list[dict], *, max_output_tokens: int):
        # LanguageModel owns protocol-specific request construction. Its public
        # return value intentionally does not expose provider token accounting.
        model = self.model if max_output_tokens == self.config.max_output_tokens else LanguageModel(
            replace(self.config, max_output_tokens=max_output_tokens), transport=self.model.transport
        )
        text, _calls, _native = model.complete(messages, [])
        return dict(text=text, usage=dict(input_tokens=0, output_tokens=0))


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def _nonempty_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _json_text(value):
    if not isinstance(value, str):
        raise ValueError("Model response text must be a string")
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped


class LanguageMutationEvolver:
    """Propose bounded numeric JSON mutations from development evidence."""

    def __init__(self, *, client: ModelClient, space: ParameterSpace,
                 max_candidates: int, max_attempts: int = 3,
                 max_output_tokens: int = 2048, temperature_note: str = "",
                 memory=None, memory_revision=None, instructions: str | None = None):
        if not callable(getattr(client, "identity", None)) or not callable(getattr(client, "complete", None)):
            raise ValueError("Declare a ModelClient")
        if not isinstance(space, ParameterSpace):
            raise ValueError("Declare a ParameterSpace")
        if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise ValueError("Declare one to 32 language candidates")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 32:
            raise ValueError("Declare one to 32 model attempts")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 1_000_000:
            raise ValueError("Declare a positive output token limit")
        if not isinstance(temperature_note, str) or not isinstance(instructions, (str, type(None))):
            raise ValueError("Temperature note and instructions must be text")
        if memory is not None and not callable(getattr(memory, "retrieve", None)):
            raise ValueError("Research memory must provide retrieve")
        if memory_revision is not None and (not isinstance(memory_revision, str) or not memory_revision):
            raise ValueError("Memory revision must be a nonempty digest")
        self.client = client
        self.space = space
        self.max_candidates = max_candidates
        self.max_attempts = max_attempts
        self.max_output_tokens = max_output_tokens
        self.temperature_note = temperature_note
        self.memory = memory
        self.memory_revision = memory_revision
        self.instructions = instructions
        self.settings = dict(max_candidates=max_candidates, max_attempts=max_attempts,
                             max_output_tokens=max_output_tokens,
                             temperature_note=temperature_note, memory_revision=memory_revision,
                             instructions=instructions)
        self._identity = deepcopy(self.identity())

    def identity(self):
        return dict(kind="language-mutation-evolver/v1", client=self.client.identity(),
                    space=self.space.identity(), settings=deepcopy(self.settings),
                    implementation=file_digest(Path(__file__)))

    def _stable(self):
        if digest(self.identity()) != digest(self._identity):
            raise ValueError("Frozen language mutation implementation changed")

    def _field_labels(self):
        fields = self.space.fields
        bases = [".".join(field.key) for field in fields]
        labels = list(bases)
        for index, label in enumerate(bases):
            if bases.count(label) > 1:
                field = fields[index]
                labels[index] = f"{field.component}:{field.file}:{label}"
        aliases = {}
        for index, field in enumerate(fields):
            label = labels[index]
            forms = {label, ".".join(field.key), "/".join(field.key),
                     f"{field.file}:{'.'.join(field.key)}",
                     f"{field.component}:{field.file}:{'.'.join(field.key)}"}
            for form in forms:
                aliases[form] = index
        return labels, aliases

    def _memory_lessons(self, context):
        if self.memory is None:
            return {}
        revision = self.memory_revision or (context or {}).get("memory_revision")
        if not revision:
            raise ValueError("Language memory requires a pinned revision")
        scope = (context or {}).get("scope", context or {})
        return self.memory.retrieve(revision, scope)

    def _evidence_rows(self, evaluations):
        rows = []
        for value in evaluations[:64]:
            trial = value.get("trial", {})
            task, case = trial.get("task"), trial.get("case", {})
            try:
                case_id = task_case_identity(task, case)
            except Exception:
                case_id = digest(dict(task=task, case=case))
            rows.append(dict(task=task, case_identity=case_id,
                             outcome=value.get("outcome"),
                             measurements=value.get("verdict", {}).get("measurements", {})))
        # Keep the complete table under four KiB, preserving row order.
        kept = []
        for row in rows:
            candidate = kept + [row]
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode()) > 4096:
                break
            kept.append(row)
        return kept

    def _prompt(self, parent, inspected, evaluations, context, lessons, errors=None):
        labels, _aliases = self._field_labels()
        fields = []
        for index, field in enumerate(self.space.fields):
            fields.append(dict(key=labels[index], component=field.component, file=field.file,
                               path=list(field.key), lower=field.lower, upper=field.upper,
                               precision=field.precision, value=inspected["values"][index]))
        body = {
            "task": "Propose bounded data-only mutations for the declared numeric fields.",
            "fields": fields,
            "development_evidence": self._evidence_rows(evaluations),
            "output": {
                "schema": {"candidates": [{"changes": {"<field key>": "<number>"},
                                             "hypothesis": "<string>",
                                             "expected_effect": "<string>"}]},
                "max_candidates": self.max_candidates,
                "rules": ["Use only declared field keys", "Numbers must be finite and within bounds",
                          "At least one value must differ from the parent", "Return JSON only, with no prose"]
            },
        }
        if self.temperature_note:
            body["temperature_note"] = self.temperature_note
        if self.instructions:
            body["additional_instructions"] = self.instructions
        if self.memory is not None:
            body["research_memory"] = [dict(id=name, hypothesis=lesson.get("hypothesis"),
                                             conditions=lesson.get("conditions"))
                                       for name, lesson in sorted(lessons.items())]
            body["output"]["schema"]["candidates"][0]["used_lessons"] = ["<lesson id>"]
        if errors:
            body["validation_errors"] = errors
        return [{"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)}]

    def _parse_and_validate(self, text, parent_values, labels, aliases, lessons):
        errors, accepted, seen = [], [], set()
        try:
            document = json.loads(_json_text(text))
        except Exception as error:
            return [], [dict(index=None, reason=f"malformed_json: {error}")]
        if not isinstance(document, dict) or not isinstance(document.get("candidates"), list):
            return [], [dict(index=None, reason="response must be an object with a candidates list")]
        for index, raw in enumerate(document["candidates"]):
            try:
                if not isinstance(raw, dict):
                    raise ValueError("candidate must be an object")
                allowed = {"changes", "hypothesis", "expected_effect", "used_lessons"}
                unknown = set(raw) - allowed
                if unknown:
                    raise ValueError("unknown candidate keys: " + ", ".join(sorted(unknown)))
                changes = raw.get("changes")
                if not isinstance(changes, dict) or not changes:
                    raise ValueError("changes must be a nonempty object")
                values = list(parent_values)
                for key, value in changes.items():
                    if key not in aliases:
                        raise ValueError(f"unknown field: {key}")
                    if not _finite_number(value):
                        raise ValueError(f"field {key} must be a finite number")
                    field = self.space.fields[aliases[key]]
                    number = float(value)
                    if not field.lower <= number <= field.upper:
                        raise ValueError(f"field {key} is outside bounds [{field.lower}, {field.upper}]")
                    values[aliases[key]] = number
                units = self.space.unit(values)
                values = self.space.quantize(units)
                if values == parent_values:
                    raise ValueError("candidate is identical to the parent after quantization")
                identity = tuple(values)
                if identity in seen:
                    raise ValueError("candidate duplicates another candidate")
                hypothesis = _nonempty_text(raw.get("hypothesis"), "hypothesis")
                expected = _nonempty_text(raw.get("expected_effect"), "expected_effect")
                used = raw.get("used_lessons", [])
                if not isinstance(used, list) or any(not isinstance(item, str) for item in used):
                    raise ValueError("used_lessons must be a list of lesson ids")
                if self.memory is None and used:
                    raise ValueError("used_lessons requires research memory")
                if self.memory is not None and any(item not in lessons for item in used):
                    raise ValueError("used_lessons contains an unknown lesson")
                seen.add(identity)
                accepted.append(dict(values=values, hypothesis=hypothesis,
                                    expected_effect=expected, used_lessons=used,
                                    source_index=index))
            except (TypeError, ValueError) as error:
                errors.append(dict(index=index, reason=str(error)))
        if len(document["candidates"]) > self.max_candidates:
            errors.extend(dict(index=index, reason=f"candidate exceeds max_candidates={self.max_candidates}")
                          for index in range(self.max_candidates, len(document["candidates"])))
        if not accepted and not errors:
            errors.append(dict(index=None, reason="response contained no candidates"))
        return accepted[:self.max_candidates], errors

    def propose(self, parent: dict, evaluations: list[dict], context: dict | None,
                output: Path) -> list[dict]:
        self._stable()
        output = Path(output).resolve()
        proposal_root = output / "language-proposal"
        frozen = proposal_root / "proposal.json"
        if frozen.exists():
            value = read_json(frozen)
            if value.get("schema") != "physicalrsi.language-proposal/v1" or not isinstance(value.get("candidates"), list):
                raise ValueError("Frozen language proposal is malformed")
            return deepcopy(value["candidates"])
        # A solved development panel needs no exploratory model call. This also
        # lets a multi-round campaign retain a solved survivor offline.
        if evaluations and all(value.get("outcome") == "success" for value in evaluations):
            atomic_json(frozen, dict(schema="physicalrsi.language-proposal/v1",
                                     parent_sha256=verify_harness(parent), attempts=0,
                                     candidates=[], costs=dict(attempts=0, input_tokens=0,
                                                               output_tokens=0)))
            return []
        inspected = self.space.inspect(parent)
        parent_sha = verify_harness(parent)
        labels, aliases = self._field_labels()
        lessons = self._memory_lessons(context)
        errors = []
        accepted = []
        attempt_records = []
        total_input = total_output = 0
        for attempt in range(1, self.max_attempts + 1):
            messages = self._prompt(parent, inspected, evaluations, context, lessons, errors or None)
            attempt_root = proposal_root / f"attempt-{attempt}"
            atomic_json(attempt_root / "prompt.json", {"messages": messages})
            result = self.client.complete(messages, max_output_tokens=self.max_output_tokens)
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                text = ""
                usage = {}
                response_error = "client response must contain text"
            else:
                text = result["text"]
                usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
                response_error = None
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            if type(input_tokens) is not int or input_tokens < 0:
                input_tokens = 0
            if type(output_tokens) is not int or output_tokens < 0:
                output_tokens = 0
            total_input += input_tokens
            total_output += output_tokens
            attempt_root.mkdir(parents=True, exist_ok=True)
            (attempt_root / "response.txt").write_text(text, encoding="utf-8")
            if response_error:
                current, current_errors = [], [dict(index=None, reason=response_error)]
            else:
                current, current_errors = self._parse_and_validate(text, inspected["values"], labels, aliases, lessons)
            errors = current_errors
            validation = dict(schema="physicalrsi.language-validation/v1", attempt=attempt,
                              accepted=[{k: v for k, v in row.items() if k != "source_index"} for row in current],
                              errors=current_errors, usage=dict(input_tokens=input_tokens, output_tokens=output_tokens))
            atomic_json(attempt_root / "validation.json", validation)
            attempt_records.append(attempt)
            if current:
                accepted = current
                break
            if attempt >= self.max_attempts:
                break
        # Materialized candidate evidence includes every proposal attempt and the
        # verified development evidence supplied to this proposer.
        attempt_evidence = {}
        for attempt in attempt_records:
            root = proposal_root / f"attempt-{attempt}"
            for name in ("prompt.json", "response.txt", "validation.json"):
                path = root / name
                attempt_evidence[str(path)] = file_digest(path)
        development_evidence = {}
        for evaluation in evaluations:
            development_evidence.update(evaluation.get("evidence", {}))
        evidence = dict(development_evidence, **attempt_evidence)
        children = []
        environment = {"scope": ((context or {}).get("scope") if isinstance((context or {}).get("scope"), str)
                                  else next((v.get("trial", {}).get("scope") for v in evaluations
                                             if isinstance(v.get("trial", {}).get("scope"), str)), "unknown"))}
        for index, proposal in enumerate(accepted[:self.max_candidates]):
            name = f"language-{parent_sha[:12]}-{attempt_records[-1]}-{index}"
            candidate, changes = self.space.candidate(parent, inspected, proposal["values"],
                                                      output=output / name, name=name)
            children.append(dict(candidate, parent_sha256=parent_sha, method="language-mutation",
                                 changes=changes, environment=deepcopy(environment), evidence=deepcopy(evidence),
                                 costs=dict(proposals=1, attempts=len(attempt_records),
                                            input_tokens=total_input, output_tokens=total_output),
                                 hypothesis=proposal["hypothesis"],
                                 expected_effect=proposal["expected_effect"], required_checks=[],
                                 used_lessons=deepcopy(proposal["used_lessons"])))
        atomic_json(frozen, dict(schema="physicalrsi.language-proposal/v1", parent_sha256=parent_sha,
                                 attempts=len(attempt_records), candidates=children,
                                 costs=dict(attempts=len(attempt_records), input_tokens=total_input,
                                            output_tokens=total_output)))
        self._stable()
        return deepcopy(children)

    def resume_proposal(self, parent: dict, evaluations: list[dict], context: dict | None,
                        output: Path) -> list[dict]:
        proposal = Path(output).resolve() / "language-proposal" / "proposal.json"
        if proposal.exists():
            value = read_json(proposal)
            if value.get("schema") != "physicalrsi.language-proposal/v1" or not isinstance(value.get("candidates"), list):
                raise ValueError("Frozen language proposal is malformed")
            return deepcopy(value["candidates"])
        return self.propose(parent, evaluations, context, output)


__all__ = ["ModelClient", "ScriptedModelClient", "HTTPModelClient", "LanguageMutationEvolver"]
