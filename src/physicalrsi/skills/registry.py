"""Content-addressed descriptors; executable providers are explicitly trusted."""
from pathlib import Path
from PhysicalRSI_core.infra.storage import digest, read_json, locked
from PhysicalRSI_core.self_harness.evaluation import freeze_json
from .protocol import CallableSkill, SkillDescriptor


class SkillRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = self.root/'skills'
        self._implementations = {}

    def register(self, skill):
        if not isinstance(skill, CallableSkill):
            raise ValueError('Register a CallableSkill')
        record = skill.identity()
        revision = digest(record)
        with locked(self.root/'.lock'):
            freeze_json(self.records/(revision+'.json'), record)
        self._implementations[revision] = skill
        return revision

    def load(self, revision, *, implementation=None):
        if not isinstance(revision, str) or len(revision) != 64 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Expected a content-addressed registry revision')
        record = read_json(self.records/(revision+'.json'))
        if digest(record) != revision:
            raise ValueError('Skill record changed')
        value = dict(record)
        if value.pop('schema') != 'physicalrsi.skill/v1':
            raise ValueError('Unknown skill schema')
        descriptor = SkillDescriptor(**value)
        provider = implementation or self._implementations.get(revision)
        if provider is not None and provider.identity() != record:
            raise ValueError('Skill implementation identity differs from the registry')
        def execute(value, context):
            if provider is None:
                raise RuntimeError('Supply a trusted implementation after restart')
            if provider.identity() != record:
                raise ValueError('Skill implementation identity changed')
            return provider.execute(value, context)
        return CallableSkill(descriptor, execute)

    def list(self):
        return [dict(self.load(path.stem).identity(), registry_revision=path.stem)
                for path in sorted(self.records.glob('*.json'))]
