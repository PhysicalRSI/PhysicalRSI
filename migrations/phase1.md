# Phase 1 migration

This change is additive: original packages, published release sources and
persistent schemas remain unchanged. No dataset, checkpoint, simulator output,
credentials or old research workspace is migrated.

The lowercase package uses `src/physicalrsi/` because the root name is occupied
by the public executable. Install with `python -m pip install -e .` or use the
repository launcher. Existing CLI arguments are delegated to `PhysicalRSI.cli`;
old Python imports remain supported. New commands are `physicalrsi study
robodyna --fixture --workspace PATH` and `physicalrsi skills --root PATH`.

The `ResearchCampaign` and `ResearchRecord` engine sources remain unchanged.
Persisted campaigns bind implementation hashes, so the facade integrates through
the existing context and evidence APIs.

## Skill registry

Registry revision identifies the complete descriptor, while its `revision`
field identifies the provider implementation. To execute after restarting:

```python
from physicalrsi.skills.registry import SkillRegistry

# Use the saved registry root, revision and matching trusted skill provider.
loaded = SkillRegistry(root).load(registry_revision, implementation=trusted_skill)
```

A mismatched provider identity is rejected. Existing Core SkillLibrary bundles
and historical Operation revisions are not converted to this registry format.
The optional SkillComposition declarations are separate from these old records.


## Verify an installation

From the repository root, install the facade and inspect both command paths:

```bash
python -m pip install -e .
./physicalrsi --plain --workspace /tmp/physicalrsi-demo-check --command '/demo'
./physicalrsi study robodyna --fixture --workspace /tmp/physicalrsi-study-check
./physicalrsi study robodyna --fixture --workspace /tmp/physicalrsi-study-check
./physicalrsi skills --root /tmp/physicalrsi-study-check/registry
```

The demo checks the existing application entry point. The study checks composed
skill execution and persisted research evidence; the repeated invocation checks
reuse of the completed campaign. For interrupted-stage recovery and registry
identity checks, run `python -m pytest -q tests/test_framework_phase1.py`.

Existing workspaces keep their original paths. Use a fresh temporary workspace
for this example and retain the full directory when archiving its evidence.
See the [framework guide](../docs/physicalrsi-framework.md) for module ownership,
environment integration and subsequent work.
