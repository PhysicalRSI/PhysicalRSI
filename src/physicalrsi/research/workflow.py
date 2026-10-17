"""Composition bindings using existing campaign and research-record interfaces."""
from pathlib import Path
from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.self_harness.evaluation import freeze_json
from PhysicalRSI_core.self_harness.research_campaign import ResearchCampaign
from PhysicalRSI_core.skill_graph.schema import normalize_composition


class ResearchWorkflow:
    """No second selection engine: bind declarations through context and evidence."""
    def __init__(self, composition):
        self.composition = normalize_composition(composition).as_dict()

    @property
    def composition_revision(self):
        return self.composition['revision']

    def campaign(self, root, *, context, **kwargs):
        scope = dict(context, composition_revision=self.composition_revision)
        return ResearchCampaign(root, context=scope, **kwargs)

    def evidence(self, path):
        path = Path(path).resolve()
        freeze_json(path, self.composition)
        return {str(path): file_digest(path)}

    def create_record(self, record, **kwargs):
        kwargs['evidence'] = dict(kwargs['evidence'], **self.evidence(record.root / 'composition.json'))
        return record.create(**kwargs)

    def bind_experiment(self, record, expected, **kwargs):
        kwargs['evidence'] = dict(kwargs['evidence'], **self.evidence(record.root / 'composition.json'))
        return record.bind_experiment(expected, **kwargs)
