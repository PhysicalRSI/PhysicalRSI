"""Stable downstream facade for the versioned PhysicalRSI core API.

Names exported here are covered by the public API's semantic versioning
policy. Existing ``PhysicalRSI_core`` module paths remain supported for
compatibility, while new downstream code can import from this single module.
"""

from PhysicalRSI_core.contracts import Contract
from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.infra import blobs
from PhysicalRSI_core.infra.blobs import BlobRef, BlobStore, is_blob_ref, refs
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator
from PhysicalRSI_core.self_harness.search_space import ParameterSpace, SearchField
from PhysicalRSI_core.self_harness.selection import (
    betting_evidence,
    cohort_order,
    comparison_identity,
    select_survivor,
    validate_admission_binding,
    validate_profile,
    write_selection,
)
from PhysicalRSI_core.self_harness.skill_mutation import SkillMutation
from .environments.gymnasium import CallablePolicy, GymnasiumEnvironment, GymnasiumVerifier

__all__ = [
    "SelfHarness",
    "ImprovementCampaign",
    "ExperimentEvaluator",
    "ExperimentRuntime",
    "Budget",
    "HarnessState",
    "TrialQuota",
    "validate_profile",
    "validate_admission_binding",
    "comparison_identity",
    "cohort_order",
    "betting_evidence",
    "select_survivor",
    "write_selection",
    "SearchField",
    "ParameterSpace",
    "SkillMutation",
    "Contract",
    "BlobRef",
    "BlobStore",
    "is_blob_ref",
    "refs",
    "blobs",
    "GymnasiumEnvironment",
    "GymnasiumVerifier",
    "CallablePolicy",
]
