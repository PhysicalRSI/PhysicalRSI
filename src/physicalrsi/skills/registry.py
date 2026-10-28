"""Logical skill records with a legacy content-addressed adapter."""
from copy import deepcopy
from pathlib import Path
from PhysicalRSI_core.infra.storage import atomic_json, canonical, read_json, locked
from .protocol import CallableSkill, CoreContext, SkillDescriptor, RECORD_SCHEMA, SCHEMA


class SkillRegistry:
    def __init__(self, root, *, registry_id=None, resolver=None):
        self.root = Path(root).resolve()
        self.records = self.root/'skills'
        self.registry_id = registry_id or self.root.name
        if not isinstance(self.registry_id, str) or not self.registry_id.strip():
            raise ValueError('Skill registry id must be a nonempty string')
        if resolver is not None and not callable(resolver):
            raise ValueError('Skill registry resolver must be callable')
        self.resolver = resolver
        self._implementations = {}

    def reference(self):
        return dict(schema='physicalrsi.skill-registry/v1', id=self.registry_id)

    def register(self, skill):
        record = skill.registry_record()
        serialized = canonical(record)
        with locked(self.root/'.lock'):
            paths = sorted(self.records.glob('*.json'))
            for path in paths:
                if canonical(read_json(path)) == serialized:
                    revision = path.stem
                    break
            else:
                numbers = [int(path.stem[7:]) for path in paths
                           if path.stem.startswith('record-') and path.stem[7:].isdigit()]
                revision = f'record-{max(numbers, default=0) + 1:06d}'
                atomic_json(self.records/(revision+'.json'), record)
        self._implementations[revision] = skill
        return revision

    def load(self, revision, *, implementation=None):
        record = read_json(self.records/(revision+'.json'))
        schema = record.get('schema') if isinstance(record, dict) else None
        if schema == SCHEMA:
            return self._load_legacy(revision, record, implementation)
        if schema != RECORD_SCHEMA:
            raise ValueError('Unknown skill schema')
        descriptor_record = record['descriptor']
        descriptor_value = dict(descriptor_record)
        descriptor_value.pop('schema', None)
        descriptor = SkillDescriptor(**descriptor_value)
        composition = None
        if record['composition'] is not None:
            from PhysicalRSI_core.skill_graph.schema import SkillComposition
            composition = SkillComposition.from_dict(record['composition'])
        provider = self._provider(revision, record, implementation)
        return _RegisteredSkill(self.root, revision, record, descriptor, provider,
                                composition=composition, registry_id=self.registry_id)

    def _provider(self, revision, record, implementation):
        provider = (implementation if implementation is not None
                    else self._implementations.get(revision))
        if provider is None and self.resolver is not None:
            provider = self.resolver(deepcopy(record), revision)
        if provider is not None and not isinstance(provider, CallableSkill):
            raise ValueError('Skill provider resolver must return a CallableSkill')
        return provider

    def load_binding(self, binding, *, implementation=None):
        """Resolve a candidate binding and confirm the saved runtime identity."""
        from PhysicalRSI_core.self_harness.skill_binding import validate_binding
        value = validate_binding(binding)
        if value.get('registry_id') is not None:
            if value['registry_id'] != self.registry_id:
                raise ValueError('Skill binding belongs to another registry')
        elif value.get("registry_root") != str(self.root):
            raise ValueError("Legacy skill binding belongs to another registry")
        revision = value.get("registry_revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Skill binding needs a registry revision")
        loaded = self.load_binding_provider(revision, implementation=implementation)
        actual = loaded.binding_identity()
        if actual.get("record") != value["record"] or actual.get("implementation") != value["implementation"]:
            raise ValueError("Resolved skill implementation differs from its binding")
        return loaded

    def load_binding_provider(self, revision, *, implementation=None):
        """Load a binding with a provider resolved for this process."""
        loaded = self.load(revision, implementation=implementation)
        if loaded.executable_identity() is None:
            raise ValueError('Skill provider resolver is required to execute this binding')
        return loaded

    def _load_legacy(self, revision, record, implementation):
        from PhysicalRSI_core.infra.storage import digest
        if digest(record) != revision:
            raise ValueError('Skill record changed')
        value = dict(record)
        value.pop('schema')
        descriptor = SkillDescriptor(**value)
        provider = self._provider(revision, record, implementation)
        if provider is not None and provider.identity() != record:
            raise ValueError('Skill implementation identity differs from the registry')

        def execute(value, context):
            if provider is None:
                raise RuntimeError('Supply a trusted implementation after restart')
            if provider.identity() != record:
                raise ValueError('Skill implementation identity changed')
            if isinstance(context, CoreContext):
                from PhysicalRSI_core.contracts import Context
                context = Context('skill-call')
            return provider.execute(value, context)

        return _RegisteredSkill(self.root, revision, record, descriptor, provider,
                                execute=execute, registry_id=self.registry_id)

    def list(self):
        return [dict(self.load(path.stem).identity(), registry_revision=path.stem)
                for path in sorted(self.records.glob('*.json'))]


class _RegisteredSkill(CallableSkill):
    """Keep the saved declaration separate from the actual runtime binding."""

    def __init__(self, root, revision, record, descriptor, provider, *, composition=None,
                 execute=None, registry_id=None):
        self._stored_record = deepcopy(record)
        self.registry_root, self.registry_revision = str(root), revision
        self.registry_id = registry_id or Path(root).name
        self._registry_bound = True
        self._provider = provider
        if execute is None:
            def execute(value, context):
                if provider is None:
                    raise RuntimeError('Supply a trusted implementation after restart')
                return provider.execute(value, context)
        super().__init__(provider.descriptor if provider is not None else descriptor, execute,
                         composition=provider.composition if provider is not None else composition,
                         executable_identity={},
                         children=provider._children if provider is not None else ())

    def stored_record(self):
        return deepcopy(self._stored_record)

    def as_operation(self):
        return self._provider.as_operation() if self._provider is not None else super().as_operation()

    def registry_record(self):
        if self._provider is None and self._stored_record['schema'] == RECORD_SCHEMA:
            return self.stored_record()
        return super().registry_record()

    def executable_identity(self):
        return self._provider.executable_identity() if self._provider is not None else None

    execution_identity = executable_identity

    def binding_identity(self):
        return dict(schema='physicalrsi.skill-binding/v2', registry_id=self.registry_id,
                    registry_revision=self.registry_revision, record=self.stored_record(),
                    implementation=self._provider.registry_record() if self._provider is not None else None)
