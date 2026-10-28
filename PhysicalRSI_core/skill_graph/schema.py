"""Versioned, data-only declarations for executable skill compositions.

The graph is a portable identity and validation boundary.  It deliberately
does not load providers or execute operations; runtimes may bind the declared
skill revisions to trusted :class:`~PhysicalRSI_core.contracts.Operation`
objects later.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from ..infra.storage import identifier


LEGACY_SCHEMA = "physicalrsi.skill-composition/v1"
SCHEMA = "physicalrsi.skill-composition/v2"
NODE_KINDS = frozenset({"primitive", "composite", "adapter"})
EDGE_KINDS = frozenset({"sequence"})


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return deepcopy(value)


@dataclass(frozen=True)
class CompositionNode:
    id: str
    kind: str
    skill: str
    revision: str
    input: dict[str, Any]
    output: dict[str, Any]
    effects: tuple[str, ...] = ()
    implementation: dict[str, Any] | None = None
    composition: dict[str, Any] | None = None

    def __post_init__(self):
        identifier(_text(self.id, "Node id"))
        if self.kind not in NODE_KINDS:
            raise ValueError("Unknown skill composition node kind")
        _text(self.skill, "Node skill")
        _text(self.revision, "Node revision")
        _object(self.input, "Node input")
        _object(self.output, "Node output")
        object.__setattr__(self, "input", deepcopy(self.input))
        object.__setattr__(self, "output", deepcopy(self.output))
        effects = tuple(self.effects)
        if any(not isinstance(value, str) or not value.strip() for value in effects):
            raise ValueError("Node effects must be nonempty strings")
        if len(set(effects)) != len(effects):
            raise ValueError("Node effects must be unique")
        object.__setattr__(self, "effects", tuple(sorted(effects)))
        object.__setattr__(self, "implementation", deepcopy(self.implementation))
        object.__setattr__(self, "composition", deepcopy(self.composition))

    def as_dict(self):
        value = {
            "id": self.id,
            "kind": self.kind,
            "skill": self.skill,
            "revision": self.revision,
            "input": deepcopy(self.input),
            "output": deepcopy(self.output),
            "effects": list(self.effects),
        }
        # Embedded declarations omit their derived revision; the enclosing
        # graph already includes their full structure in its own revision.
        if self.implementation is not None:
            value["implementation"] = deepcopy(self.implementation)
        if self.composition is not None:
            value["composition"] = deepcopy(self.composition)
        return value


@dataclass(frozen=True)
class CompositionEdge:
    source: str
    target: str
    kind: str = "sequence"
    condition: str | None = None

    def __post_init__(self):
        identifier(_text(self.source, "Edge source"))
        identifier(_text(self.target, "Edge target"))
        if self.source == self.target:
            raise ValueError("A composition edge cannot point to itself")
        if self.kind not in EDGE_KINDS:
            raise ValueError("Unknown skill composition edge kind")
        if self.condition is not None:
            raise ValueError("Conditional execution is not supported in Phase 1")

    def as_dict(self):
        value = {"from": self.source, "to": self.target, "kind": self.kind}
        if self.condition is not None:
            value["condition"] = self.condition
        return value


@dataclass(frozen=True)
class SkillComposition:
    id: str
    ontology_revision: str
    embodiment_profile: str
    entry: str
    exit: str
    nodes: tuple[CompositionNode, ...]
    edges: tuple[CompositionEdge, ...]
    qualification: dict[str, Any] | None = None
    _schema: str = field(default=SCHEMA, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_schema", SCHEMA)
        identifier(_text(self.id, "Composition id"))
        _text(self.ontology_revision, "Ontology revision")
        _text(self.embodiment_profile, "Embodiment profile")
        identifier(_text(self.entry, "Composition entry"))
        identifier(_text(self.exit, "Composition exit"))
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if not nodes:
            raise ValueError("A skill composition needs at least one node")
        if any(not isinstance(node, CompositionNode) for node in nodes):
            raise ValueError("Composition nodes must be CompositionNode values")
        if any(not isinstance(edge, CompositionEdge) for edge in edges):
            raise ValueError("Composition edges must be CompositionEdge values")
        ids = [node.id for node in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("Composition node ids must be unique")
        if self.entry not in ids or self.exit not in ids:
            raise ValueError("Composition entry and exit must name nodes")
        known = set(ids)
        if any(edge.source not in known or edge.target not in known for edge in edges):
            raise ValueError("Composition edge refers to an unknown node")
        if len(edges) != len(nodes)-1:
            raise ValueError("Phase 1 requires a serial graph")
        successors = {}
        predecessors = {}
        by_id = {node.id: node for node in nodes}
        for edge in edges:
            if edge.source in successors or edge.target in predecessors:
                raise ValueError("Phase 1 requires a serial graph")
            successors[edge.source] = edge.target
            predecessors[edge.target] = edge.source
        visited = set()
        cursor = self.entry
        while cursor not in visited:
            visited.add(cursor)
            if cursor not in successors:
                break
            cursor = successors[cursor]
        if visited != known or cursor != self.exit or self.exit in successors:
            raise ValueError("Composition must be one connected acyclic sequence")
        if self.qualification is not None:
            _object(self.qualification, "Qualification")
        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda node: node.id)))
        object.__setattr__(self, "edges", tuple(sorted(
            edges, key=lambda edge: (edge.source, edge.target, edge.kind, edge.condition or "")
        )))

    def as_dict(self, *, include_revision=True):
        value = {
            "schema": self._schema,
            "id": self.id,
            "ontology_revision": self.ontology_revision,
            "embodiment_profile": self.embodiment_profile,
            "entry": self.entry,
            "exit": self.exit,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "qualification": deepcopy(self.qualification),
        }
        if include_revision:
            value["revision"] = self.revision
        return value

    @property
    def revision(self):
        if self._schema == LEGACY_SCHEMA:
            from ..infra.storage import digest
            return digest(self.as_dict(include_revision=False))
        return json.dumps(
            self.as_dict(include_revision=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping) or value.get("schema") not in {LEGACY_SCHEMA, SCHEMA}:
            raise ValueError(f"Expected a {SCHEMA} declaration")
        required = {
            "schema", "id", "revision", "ontology_revision", "embodiment_profile",
            "entry", "exit", "nodes", "edges", "qualification",
        }
        if set(value) != required:
            raise ValueError("Skill composition has unknown or missing fields")
        if not isinstance(value["nodes"], list) or not isinstance(value["edges"], list):
            raise ValueError("Composition nodes and edges must be lists")
        nodes = []
        for row in value["nodes"]:
            required_node = {
                "id", "kind", "skill", "revision", "input", "output", "effects"
            }
            if (not isinstance(row, Mapping) or not required_node <= set(row)
                    or set(row) - required_node - {"implementation", "composition"}):
                raise ValueError("Composition node has unknown or missing fields")
            nodes.append(CompositionNode(**dict(row)))
        edges = []
        for row in value["edges"]:
            if not isinstance(row, Mapping) or set(row) - {"from", "to", "kind", "condition"} \
                    or not {"from", "to"} <= set(row):
                raise ValueError("Composition edge has unknown or missing fields")
            edges.append(CompositionEdge(
                source=row["from"], target=row["to"], kind=row.get("kind", "sequence"),
                condition=row.get("condition")))
        result = cls(
            id=value["id"], ontology_revision=value["ontology_revision"],
            embodiment_profile=value["embodiment_profile"], entry=value["entry"],
            exit=value["exit"], nodes=tuple(nodes), edges=tuple(edges),
            qualification=value["qualification"],
        )
        if value.get("schema") == LEGACY_SCHEMA:
            from ..infra.storage import digest
            legacy = result.as_dict(include_revision=False)
            legacy["schema"] = LEGACY_SCHEMA
            expected = digest(legacy)
            object.__setattr__(result, "_schema", LEGACY_SCHEMA)
        else:
            expected = result.revision
        if value["revision"] != expected:
            raise ValueError("Skill composition revision differs from its content")
        return result


def normalize_composition(value) -> SkillComposition:
    """Accept a declaration or an already validated composition object."""
    if isinstance(value, SkillComposition):
        return value
    attached = getattr(value, "composition", None)
    if isinstance(attached, SkillComposition):
        return attached
    if attached is not None:
        value = attached
    return SkillComposition.from_dict(value)
