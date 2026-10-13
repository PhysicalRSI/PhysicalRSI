"""One task-neutral Trial/Evaluation contract over verified runtime evidence.

JSON documents are privately serialized so caller mutation cannot change an
existing value. A completed evaluation preserves all four native outcomes;
scoring and survivor selection belong to Self-Harness, not this contract.
"""
from dataclasses import asdict, dataclass
import json

from .embodiment import Revision, System2Trial
from .experiments import Budget
from .infra.storage import canonical, digest, identifier


def task_case_identity(task, case):
    """Execution-case identity; seed namespaces belong to individual tasks.

    Semantic reset identity (ignoring auxiliary metadata) is separately enforced
    by CaseProtocol. Task names must identify real task domains, not aliases
    invented to bypass validation reuse checks.
    """
    if not isinstance(task, str) or not task or not isinstance(case, dict):
        raise ValueError("Declare a task and its public case")
    return digest(dict(task=task, case=case))


@dataclass(frozen=True, init=False)
class Trial:
    _document: str

    def __init__(self, run_id, *, task, case, scope, budget=None, system2=None, replacement=None):
        identifier(run_id)
        if any(not isinstance(value, str) or not value.strip() for value in (task, scope)):
            raise ValueError("Trial requires task and scope")
        if not isinstance(case, dict):
            raise ValueError("Trial case must be a public JSON object")
        budget = budget if budget is not None else Budget()
        if not isinstance(budget, Budget) or (system2 is not None and not isinstance(system2, System2Trial)):
            raise ValueError("Invalid trial budget or System 2 attribution")
        value = dict(schema="physicalrsi.trial/v1",
            id=run_id, task=task, case=case, scope=scope, budget=asdict(budget),
            system2=asdict(system2) if system2 is not None else None)
        if replacement is not None:
            value['replacement'] = replacement
        object.__setattr__(self, "_document", canonical(value).decode())

    def to_dict(self):
        return json.loads(self._document)

    @property
    def revision(self):
        return digest(self.to_dict())

    def arguments(self):
        value = self.to_dict()
        value.pop("schema")
        value["run_id"] = value.pop("id")
        value["budget"] = Budget(**value["budget"])
        if value["system2"] is not None:
            origin = value["system2"]
            origin["system2"] = Revision(**origin["system2"])
            value["system2"] = System2Trial(**origin)
        return value


@dataclass(frozen=True, init=False)
class Evaluation:
    _document: str

    @classmethod
    def read(cls, runtime, run_id):
        audited = runtime.read_evidence(run_id)
        receipt, spec = audited["receipt"], audited["specification"]
        origin = spec.get("system2")
        if origin is not None:
            origin = dict(origin, system2=Revision(**origin["system2"]))
            origin = System2Trial(**origin)
        trial = Trial(run_id, task=spec["task"], case=spec["case"], scope=spec["scope"],
                      budget=Budget(**spec["budget"]), system2=origin, replacement=spec.get('replacement'))
        documents, evidence = audited["documents"], audited["evidence"]
        value = dict(schema="physicalrsi.evaluation/v1", trial=trial.to_dict(),
                     trial_sha256=trial.revision, outcome=receipt["outcome"],
                     qualification=receipt["qualification"], reset=documents["reset.json"],
                     trajectory=documents["trajectory.json"], verdict=documents["verdict.json"],
                     receipt=receipt, evidence=evidence)
        if 'recovery' in audited:
            value['recovery'] = audited['recovery']
        result = object.__new__(cls)
        object.__setattr__(result, "_document", canonical(value).decode())
        return result

    def to_dict(self):
        return json.loads(self._document)

    @property
    def outcome(self):
        return self.to_dict()["outcome"]

    @property
    def revision(self):
        return digest(self.to_dict())
