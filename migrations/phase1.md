# Phase 1 migration

This change is additive: original packages, published release sources and
persistent schemas remain unchanged. No dataset, checkpoint, simulator output,
credentials or old research workspace is migrated.

The lowercase package uses `src/physicalrsi/` because the root name is occupied
by the public executable. Install with `python -m pip install -e .` or use the
repository launcher. Existing CLI arguments are delegated to `PhysicalRSI.cli`;
old Python imports remain supported. New commands are `physicalrsi study
robodyna --fixture --workspace PATH` and `physicalrsi skills --root PATH`.

Previously proposed direct edits to `ResearchCampaign` and `ResearchRecord`
were removed before this migration: changing engine implementation hashes can
invalidate persisted campaigns. The facade uses existing context/evidence APIs.

Registry revision identifies the complete descriptor, while its `revision`
field identifies the provider implementation. To execute after restarting:

```python
loaded = SkillRegistry(root).load(registry_revision, implementation=trusted_skill)
```

A mismatched provider identity is rejected. Existing Core SkillLibrary bundles
and historical Operation revisions are not converted to this registry format.
The optional SkillComposition declarations are separate from these old records.
