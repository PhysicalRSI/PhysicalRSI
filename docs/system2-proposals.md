# Structured System 2 proposals

`StructuredProposer` turns development evidence into separately materialized JSON configuration candidates. A reviewed finite search (`EnumeratedEdits`) and an external HTTP model (`ModelProposal`) use the same edit contract, request/response journal, candidate admission, paired experiments and lineage selection. Changing the strategy does not change System 1's action path or give the strategy control over the verifier.

This follows the separation of proposals, reproducible attempts and independently evaluated artifacts described in OpenRSI's [pinned overview](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Evo/docs/overview.md) and [run guide](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Evo/docs/usage.md). These modules are an independent implementation for the physicalRSI experiment boundary.

## The two roles

| Decision | Owner |
| --- | --- |
| Which observation, skill, tool or memory to use during an episode | Frozen System 1 policy |
| Which declared configuration edits to propose between episodes | System 2 strategy |
| Which paths and JSON values are eligible for change | Task author's `JsonEditContract` |
| Whether a candidate's semantics and dependency closure are admissible | Experiment suite admission |
| Which independent outcomes count as improvement | Frozen verifier, paired evaluator and selector |
| Which revision becomes the next parent | Self-Harness and lineage compare-and-swap |

The strategy port is trusted host code with `identity()` and `generate(request, limits)`. A remote model's reply is untrusted data. The built-in model adapter executes no model-selected tools or commands. `IsolatedProposal` provides an optional [restricted Linux worker](isolated-candidates.md) when the strategy implementation itself is untrusted Python source. It returns the same data-only reply and has no network or device callbacks. The default examples edit configuration. Suites can explicitly admit Python skill artifacts using `PYTHON_SKILL_SCHEMA` and execute them through `IsolatedHarnessPolicy`; this binds source edits to the candidate closure and restricted System 1 runtime. JSON materialization itself never executes source. Checkpoint training and automatic changes to the foundation remain outside the edit contract.

## Declare an edit domain

```python
from PhysicalRSI_core.self_harness.proposals import (
    EnumeratedEdits, JsonEditContract, ProposalLimits, StructuredProposer,
)

contract = JsonEditContract({
    "control.json": {
        "component": "control",
        "schema": {
            "type": "object",
            "properties": {"mode": {"enum": ["zero-effort", "feedback"]}},
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
})
strategy = EnumeratedEdits([{"control.json": {"mode": "feedback"}}])
# evaluator is a configured ExperimentEvaluator for the task suite.
proposer = StructuredProposer(
    evaluator=evaluator,
    strategy=strategy,
    contract=contract,
    objective="Improve measured position and velocity success on both fixtures.",
    limits=ProposalLimits(max_candidates=1, seconds=120),
)
```

Pass this proposer to the existing `SelfHarness`; its candidate allowance must cover the proposal limit. The parent and every edited value must satisfy the schema. Schemas are self-contained JSON Schema 2020-12, without references that could trigger external reads. Editable files must already exist in the verified harness and belong to exactly their named component. Foundation files, aliases, absolute paths, traversal, file creation and deletion are rejected. Other JSON components, including declared memory rules or skill routing configuration, can use the same mechanism when the task suite admits them.

The request contains the objective, parent/component digests, editable file values and schemas, development episode summaries and evidence digests. Evidence files are verified before a call. The adapter does not include the parent workspace path, undeclared files, complete trajectories or validation cases. Task authors must supply development summaries and editable configuration appropriate for their chosen provider; this projection is not a general secret classifier or an information-isolation sandbox.

A response contains only `parent_sha256` and a `candidates` list. Each candidate contains `rationale` and `edits`; each edit contains `path`, `before_sha256` and `value`. Additional self-reported performance fields are rejected. Duplicate JSON members, nonfinite numbers, stale file revisions, out-of-schema values and repeated edit paths are rejected. Invalid responses produce a recorded rejection and no child. Identical JSON values and duplicate executable candidates do not consume validation trials.

Valid children copy only the declared harness files into separate directories, apply the allowed edits, retain the foundation digest and record their parent, strategy, changes, evidence and shared proposal cost. The task's existing admission gate still checks semantic safety and dependency completeness before any candidate runs. A JSON schema alone is not a robot safety argument.

## Native simulation with the shared proposer

```bash
pip install -e '.[mujoco]'
study_dir="$(mktemp -d /tmp/physicalrsi-structured-proposal.XXXXXX)"
python -m PhysicalRSI_demos.mujoco_evolution \
  --workspace "$study_dir" --proposer enumerated --rounds 2 --trials 12

# Audit the same immutable campaign; no new strategy call or trial is needed.
python -m PhysicalRSI_demos.mujoco_evolution \
  --workspace "$study_dir" --proposer enumerated --rounds 2 --trials 12
```

This exercises the common proposal path against native MuJoCo slider and hinge dynamics, through controller ownership, independent verification and actual lineage inheritance. The sole search alternative enables an existing reviewed feedback controller; it is not a learned controller. The next round sees the selected value and proposes no change. The default `--proposer reviewed` retains the earlier task-specific deterministic repair example. Both demonstrations remain simulation-only with `qualification: null`.

## Optional external model

Replace the strategy with `ModelProposal(ModelConfig(...), provider_revision="...")`, or use `--proposer model` in the demo. Supply a JSON file containing the existing `ModelConfig` fields:

```json
{
  "model": "YOUR_PROVIDER_MODEL",
  "base_url": "https://YOUR_PROVIDER_API_ROOT/v1",
  "api_key_env": "PHYSICALRSI_API_KEY",
  "protocol": "responses",
  "timeout_s": 60,
  "max_output_tokens": 4096,
  "max_tool_rounds": 1
}
```

`chat_completions` is also supported through the existing language boundary. Configure the named environment variable using your normal credential mechanism; the JSON file contains its name, not its value. Then run:

```bash
study_dir="$(mktemp -d /tmp/physicalrsi-model-proposal.XXXXXX)"
python -m PhysicalRSI_demos.mujoco_evolution \
  --workspace "$study_dir" --proposer model \
  --model-config /path/to/model-config.json \
  --provider-revision YOUR_PINNED_PROVIDER_REVISION \
  --rounds 2 --trials 12
```

The supplied provider revision is a declaration, not verification of external weights. The adapter sends one tool-free request per proposal round. It preserves the provider's response ID, model name and reported usage when returned. The [OpenAI Responses reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) describes the response and usage fields used by that protocol; other compatible providers must be checked for their own behavior. No token prices, training cost or successful inference are inferred from those fields. Absent usage stays `null`.

The automated HTTP tests use a local fixture provider with deterministic JSON replies. They demonstrate transport, accounting, rejection and integration behavior, not language-model capability. Native simulation tests use the reviewed and enumerated strategies. A real external model's repair quality has not been established by these tests.

## Limits and recovery

`ProposalLimits` freezes candidate count, request/response byte limits, per-candidate copied closure size and model-call seconds. The default byte limits are 256 KiB for the public request and response, and 64 MiB for each copied closure. Request limits describe the canonical task payload; protocol framing adds a fixed instruction and model configuration. Large external dependencies should stay behind separately pinned and verified asset manifests.

The model adapter runs the existing language protocol in an owned worker process. The total call deadline includes startup, DNS, connection, TLS and reading a trickling response. Teardown has a separate bounded allowance. Redirects are not followed with credentials. The worker has no tool execution loop; a tool call or incomplete response cannot create a candidate. Custom strategy implementations must enforce their own compute limits; the port itself is not a sandbox or a whole-campaign deadline.

Each round reserves one strategy call before invoking it. A campaign's fixed round count therefore bounds this adapter's call count, separately from `TrialQuota` and the per-experiment action budget. All candidates from one reply share that call cost; summing their provenance would double count it. Output tokens are requested through `ModelConfig.max_output_tokens`; usage remains provider-reported. Timeout does not establish provider cancellation or zero billing. There is no global multi-campaign model spending ledger.

```text
round/
  proposal/request.json          frozen public input projection
  proposal/call.json             started intent or completed response and usage
  proposal/candidates/<digest>/  separately materialized declared files
  proposal/materialization.json accepted children and rejection reasons
  propose.json                   outer Self-Harness stage receipt
```

The completed reply is durable before parsing or copying candidates. `resume_proposal` can finish materialization from that same reply after interruption, verify an already written child, and finish the outer checkpoint without another model request. Changed inputs, response digests, changed candidate files or additional undeclared files fail verification. Once validation is exposed, a new proposal call in that round is forbidden.

A started call without a durable reply requires reconciliation and is never sent again automatically, even if the client might have failed before reaching the provider. Arbitrary provider exceptions are recorded by type rather than raw text. A known invalid reply is retained as rejection evidence and also is not retried. These are at-most-once automatic attempt semantics, not a claim of exactly-once remote execution. There is no built-in operator override for an unknown model call; preserve that study and explicitly choose a new study after investigating the provider-side outcome.
