"""Owned harness branches with deterministic routing from public capabilities.

A tree is one ordinary candidate, not a second deployment authority. Every
branch file remains in its original component, including logical file names.
The same admission, paired evaluation, selection and rollback apply to it.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import tempfile

from .embodiment import Dependency, System1
from .infra.owned_files import OwnedFiles, physical
from .infra.storage import atomic_json, digest, file_digest, identifier, read_json, relative_path
from .self_harness.artifacts import CLOSURE, EvolutionSpec, bind_harness_modes, harness_identity, verify_harness


def _object_path(kind, sha, mode=None):
    return 'objects/' + kind + '/' + sha + ('' if mode is None else '-' + format(mode, '03o'))


def _shape(nodes, keys, root):
    if (not isinstance(keys, list) or keys != sorted(set(keys)) or "task" not in keys
            or any(not isinstance(k, str) or not k for k in keys)):
        raise ValueError("Declare unique capability keys including task")
    if not isinstance(nodes, dict) or not 1 <= len(nodes) <= 64 or root not in nodes:
        raise ValueError("A harness tree requires 1 to 64 nodes and one root")
    children = {name: [] for name in nodes}
    for name, row in nodes.items():
        identifier(name)
        when = row["when"]
        if (not isinstance(when, dict) or not set(when) <= set(keys)
                or any(not isinstance(v, str) or not v for v in when.values())):
            raise ValueError("Routes require declared exact string capabilities")
        if name == root:
            if row["parent"] is not None or when:
                raise ValueError("The root must be an unconditional fallback")
        else:
            parent = row["parent"]
            if parent not in nodes or parent == name:
                raise ValueError("Unknown or cyclic harness parent")
            prior = nodes[parent]["when"]
            if len(when) <= len(prior) or any(when.get(k) != v for k, v in prior.items()):
                raise ValueError("A child must strictly narrow its parent's capabilities")
            children[parent].append(name)
    for names in children.values():
        for index, left in enumerate(names):
            a = nodes[left]["when"]
            for right in names[index + 1:]:
                b = nodes[right]["when"]
                if not any(k in b and b[k] != v for k, v in a.items()):
                    raise ValueError("Sibling capability routes overlap")
    pending, seen = [(root, 0)], set()
    while pending:
        name, depth = pending.pop()
        if name in seen or depth > 16:
            raise ValueError("Cyclic or excessively deep harness tree")
        seen.add(name)
        pending.extend((child, depth + 1) for child in children[name])
    if seen != set(nodes):
        raise ValueError("Disconnected harness tree")
    return children


class HarnessTree:
    @staticmethod
    def create(nodes, *, keys, output, name, evolution=None):
        """Atomically package complete source harnesses as a single candidate."""
        identifier(name)
        evolution = evolution or EvolutionSpec()
        nodes, keys = deepcopy(nodes), sorted(keys)
        roots = [name for name, row in nodes.items() if row["parent"] is None]
        if len(roots) != 1:
            raise ValueError("Declare exactly one harness root")
        root = roots[0]
        _shape(nodes, keys, root)
        output = Path(output).resolve()
        anchor = nodes[root]["harness"]
        bound_modes = 'file_modes' in anchor
        for row in nodes.values():
            evolution.compare(anchor, row["harness"])
            source = Path(row["harness"]["root"]).resolve()
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("Package trees outside source harnesses")
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".compose-", dir=output))
        try:
            components, exports, records = {kind: {} for kind in CLOSURE}, {}, {}
            for node, row in sorted(nodes.items()):
                references = {}
                for kind, files in sorted(row["harness"]["components"].items()):
                    reference = "manifests/" + kind + "/" + digest(files) + ".json"
                    atomic_json(temporary / reference, files)
                    sha = file_digest(temporary / reference)
                    exports[reference] = dict(path=str(temporary / reference), sha256=sha)
                    components[kind][reference] = sha
                    references[kind] = reference
                    for logical, sha in files.items():
                        path = physical(Path(row["harness"]["root"]), logical)
                        mode = row['harness'].get('file_modes', {}).get(logical)
                        target = _object_path(kind, sha, mode)
                        exports[target] = dict(path=str(path), sha256=sha)
                        components[kind][target] = sha
                records[node] = dict(parent=row["parent"], when=row["when"], components=references)
                if bound_modes:
                    records[node]['file_modes'] = deepcopy(row['harness']['file_modes'])
            tree = dict(schema="physicalrsi.harness-tree/v2" if bound_modes else "physicalrsi.harness-tree/v1", root=root, keys=keys, nodes=records,
                        evolution=evolution.identity(), implementation=file_digest(Path(__file__)))
            atomic_json(temporary / "tree.json", tree)
            sha = file_digest(temporary / "tree.json")
            components["skill_selection"]["tree.json"] = sha
            exports["tree.json"] = dict(path=str(temporary / "tree.json"), sha256=sha)
            owned = OwnedFiles(output)
            revision = owned.capture(dict(schema="physicalrsi.tree-package/v1", components=components), exports)
            candidate = dict(id=name, root=str(owned.folder(revision) / "files"), components=components)
            if bound_modes:
                candidate = bind_harness_modes(candidate)
            HarnessTree(candidate)
            return candidate
        finally:
            shutil.rmtree(temporary)

    def __init__(self, candidate):
        self.candidate = deepcopy(candidate)
        self.revision = verify_harness(candidate)
        self.root = Path(candidate["root"])
        body = read_json(physical(self.root, "tree.json"))
        if (set(body) != {"schema", "root", "keys", "nodes", "evolution", "implementation"}
                or body["schema"] not in {"physicalrsi.harness-tree/v1", "physicalrsi.harness-tree/v2"}
                or body["implementation"] != file_digest(Path(__file__))):
            raise ValueError("Unknown or changed harness tree implementation")
        self.children = _shape(body["nodes"], body["keys"], body["root"])
        evolution = EvolutionSpec.from_identity(body["evolution"])
        bound_modes = body['schema'].endswith('/v2')
        if bound_modes != ('file_modes' in candidate):
            raise ValueError('Tree and candidate file mode declarations differ')
        components, leaves = {kind: {} for kind in CLOSURE}, {}
        components["skill_selection"]["tree.json"] = file_digest(self.root / "tree.json")
        for node, row in body["nodes"].items():
            if set(row) != {"parent", "when", "components", *({'file_modes'} if bound_modes else set())} or set(row["components"]) != CLOSURE:
                raise ValueError("Incomplete tree node closure")
            leaf, aliases = {}, {}
            for kind, reference in row["components"].items():
                files = read_json(physical(self.root, reference))
                if not isinstance(files, dict) or not files or reference != "manifests/" + kind + "/" + digest(files) + ".json":
                    raise ValueError("Tree component manifest changed")
                components[kind][reference] = file_digest(self.root / reference)
                for logical, sha in files.items():
                    if str(relative_path(logical)) != logical or (logical in aliases and aliases[logical] != sha):
                        raise ValueError("Conflicting or noncanonical branch file names")
                    aliases[logical] = sha
                    mode = row.get('file_modes', {}).get(logical)
                    path = _object_path(kind, sha, mode)
                    if file_digest(physical(self.root, path)) != sha:
                        raise ValueError("Tree branch bytes changed")
                    if bound_modes and candidate['file_modes'][path] != mode:
                        raise ValueError('Tree branch file mode differs from its declaration')
                    components[kind][path] = sha
                leaf[kind] = files
            leaves[node] = leaf
            if bound_modes:
                harness_identity(dict(components=leaf, file_modes=row['file_modes']))
        for node, leaf in leaves.items():
            if any(leaf[kind] != leaves[body["root"]][kind] for kind in CLOSURE - set(evolution.mutable)):
                raise ValueError("Branch changed a fixed component")
            if bound_modes and any(body['nodes'][node]['file_modes'][n] != body['nodes'][body['root']]['file_modes'][n]
                    for kind in CLOSURE - set(evolution.mutable) for n in leaf[kind]):
                raise ValueError('Branch changed a fixed component file mode')
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() or p.is_symlink()}
        if (components != candidate["components"]
                or actual != {p for files in components.values() for p in files}):
            raise ValueError("Tree closure differs from its declared components")
        self.body, self.leaves = body, leaves

    def leaf_identity(self, node):
        value = dict(components=deepcopy(self.leaves[node]))
        if 'file_modes' in self.body['nodes'][node]:
            value['file_modes'] = deepcopy(self.body['nodes'][node]['file_modes'])
        return harness_identity(value)

    def route(self, capabilities):
        if (not isinstance(capabilities, dict) or set(capabilities) != set(self.body["keys"])
                or any(not isinstance(v, str) or not v for v in capabilities.values())):
            raise ValueError("Routing requires the exact public capability schema")
        path = [self.body["root"]]
        while True:
            matches = [n for n in self.children[path[-1]] if all(capabilities[k] == v
                       for k, v in self.body["nodes"][n]["when"].items())]
            if not matches:
                return path
            if len(matches) != 1:
                raise ValueError("Ambiguous runtime route")
            path.append(matches[0])

    def materialize(self, node, output):
        """Reconstruct the original logical file layout for an existing backend."""
        if HarnessTree(self.candidate).revision != self.revision:
            raise ValueError("Tree changed before materialization")
        owned = OwnedFiles(output)
        if owned.root.is_relative_to(self.root) or self.root.is_relative_to(owned.root):
            raise ValueError("Materialize outside the frozen tree")
        components = self.leaves[node]
        modes = self.body['nodes'][node].get('file_modes', {})
        exports = {name: dict(path=str(self.root / _object_path(kind, sha, modes.get(name))), sha256=sha)
                   for kind, files in components.items() for name, sha in files.items()}
        revision = owned.capture(dict(schema="physicalrsi.tree-leaf/v1", components=components), exports)
        candidate = dict(id=node, root=str(owned.folder(revision) / "files"), components=deepcopy(components))
        if 'file_modes' in self.body['nodes'][node]:
            candidate['file_modes'] = deepcopy(modes)
        verify_harness(candidate)
        return candidate


class RoutedPolicy:
    """Pin one branch per episode from the initial public observation.

    Backend.build(harness, task) constructs ordinary declared policies without
    starting workers. Extractor.extract(observation) sees no private case or
    verifier data. Both ports are explicitly supplied trusted implementations.
    """
    def __init__(self, candidate, *, task, backend, extractor, workspace):
        self.tree, self.task = HarnessTree(candidate), task
        self.backend, self.extractor = backend, extractor
        self.backend_identity, self.extractor_identity = deepcopy(backend.identity()), deepcopy(extractor.identity())
        self.leaves, self.policies = {}, {}
        for node, row in self.tree.body["nodes"].items():
            if row["when"].get("task", task) != task:
                continue
            leaf = self.tree.materialize(node, workspace)
            policy = backend.build(deepcopy(leaf), task)
            description = policy.describe()
            if not isinstance(description, System1) or description.revision != verify_harness(leaf):
                raise ValueError("Branch policy differs from its executable closure")
            if (not callable(getattr(policy, "begin_episode", None))
                    or not callable(getattr(policy, "act", None))
                    or any(getattr(policy, method, None) is not None and not callable(getattr(policy, method))
                           for method in ("end_episode", "execution_evidence"))):
                raise ValueError("Invalid branch policy lifecycle")
            self.leaves[node], self.policies[node] = leaf, policy
        descriptions = {node: policy.describe() for node, policy in self.policies.items()}
        base = descriptions[self.tree.body["root"]]
        if any((d.observation, d.action, d.timing) != (base.observation, base.action, base.timing)
               for d in descriptions.values()):
            raise ValueError("Routed branches require matching task contracts and timing")
        dependencies = tuple(Dependency(node + "." + d.name, d.revision, d.kind)
                             for node, spec in descriptions.items() for d in spec.dependencies)
        self.description = replace(base, name="harness-tree", revision=self.tree.revision, dependencies=dependencies)
        self._identity = dict(kind="physicalrsi.routed-policy/v1", tree=self.tree.revision, task=task,
            backend=self.backend_identity, extractor=self.extractor_identity,
            branches={node: dict(identity=policy.identity(), binding=asdict(descriptions[node]),
                                lifecycle=callable(getattr(policy, "end_episode", None)),
                                evidence=callable(getattr(policy, "execution_evidence", None)))
                      for node, policy in self.policies.items()}, implementation=file_digest(Path(__file__)))
        # Cleanup must still reach the started worker if its files or public
        # lifecycle attribute change. It does not authorize another action.
        self._cleanup = {node:getattr(policy, "end_episode", None) for node,policy in self.policies.items()}
        self.selected, self.route_evidence = None, None

    def identity(self):
        if (self.backend.identity() != self.backend_identity or self.extractor.identity() != self.extractor_identity
                or file_digest(Path(__file__)) != self._identity["implementation"]
                or file_digest(physical(self.tree.root, "tree.json")) != self.tree.candidate["components"]["skill_selection"]["tree.json"]):
            raise ValueError("Frozen tree or routing provider changed")
        for node, policy in self.policies.items():
            if ((self.selected is None or node == self.selected)
                    and verify_harness(self.leaves[node]) != digest(self.tree.leaf_identity(node))):
                raise ValueError("Materialized branch files changed")
            if (dict(identity=policy.identity(), binding=asdict(policy.describe()),
                            lifecycle=callable(getattr(policy, "end_episode", None)),
                            evidence=callable(getattr(policy, "execution_evidence", None))) != self._identity["branches"][node]):
                raise ValueError("Materialized branch policy changed")
        return deepcopy(self._identity)

    def describe(self):
        return self.description

    def _context(self, context):
        if context.harness_revision != self.tree.revision:
            raise ValueError("Routing execution belongs to a different candidate")
        return replace(context, harness_revision=self._identity["branches"][self.selected]["binding"]["revision"])

    def begin_episode(self, task, case, observation, context):
        self.identity()
        if task != self.task or context.harness_revision != self.tree.revision:
            raise ValueError("Routed episode belongs to a different task or candidate")
        capabilities = self.extractor.extract(deepcopy(observation))
        if not isinstance(capabilities, dict) or "task" in capabilities:
            raise ValueError("Extractor cannot replace the runtime task")
        capabilities = dict(capabilities, task=task)
        path = self.tree.route(capabilities)
        self.selected = path[-1]
        self.route_evidence = dict(schema="physicalrsi.route-evidence/v1", tree_revision=self.tree.revision,
            capabilities=deepcopy(capabilities), path=path, leaf_revision=verify_harness(self.leaves[self.selected]),
            extractor=self.extractor_identity)
        self.policies[self.selected].begin_episode(task, deepcopy(case), deepcopy(observation), self._context(context))

    def act(self, observation, context):
        self.identity()
        if self.selected is None:
            raise ValueError("Begin an episode before routing actions")
        return self.policies[self.selected].act(observation, self._context(context))

    def end_episode(self, context):
        if self.selected is None:
            return dict(stopped=True, started=False)
        method = self._cleanup[self.selected]
        report = method(self._context(context)) if callable(method) else dict(stopped=True)
        return dict(report, routed_node=self.selected)

    def execution_evidence(self):
        self.identity()
        # Full-tree audits bracket execution. On individual actions identity()
        # checks the pinned routing and the executing leaf, without rehashing
        # every inactive branch on each runtime liveness check.
        HarnessTree(self.tree.candidate)
        for node, leaf in self.leaves.items():
            if verify_harness(leaf) != digest(self.tree.leaf_identity(node)):
                raise ValueError("Materialized branch files changed")
        if self.route_evidence is None:
            return dict(started=False)
        value = deepcopy(self.route_evidence)
        method = getattr(self.policies[self.selected], "execution_evidence", None)
        if callable(method):
            value["branch_evidence"] = deepcopy(method())
        return value
