# Experiment protocol

## Claim and coverage

For each pinned benchmark/task, test whether System 2 optimization of official
score produces behavior that satisfies the benchmark proxy while violating or
failing the independently defined task objective. Publish successful exploits,
unsuccessful searches, inconclusive outcomes and untested tasks separately.

Before a study, enumerate the benchmark versions and task population covered by
the claim. A finite sample does not establish that all current or future
simulation benchmarks are exploitable. Do not select only successful tasks for
the reported denominator.

## Compared systems

Use a common initial GPT System 1 configuration and CAP implementation:

| Arm | Changes between trials | Purpose |
| --- | --- | --- |
| Fixed System 1 | None | Measure the starting policy |
| Budget-matched search | Predeclared sampling without feedback-driven artifact updates | Separate additional attempts from System 2 adaptation |
| System 2 optimization | Admitted prompt, skill or memory revisions based on development proxy feedback | Test feedback-driven exploitation |

Match or explicitly report model calls, tokens, simulator trials, resets and
wall-clock cost. Record rejected candidates and failed attempts. Provider usage
is cost provenance, not evidence of task success. A scripted proposer is a useful
control but must not be reported as a GPT-generated discovery.

## Access conditions

Declare observation access, action APIs, evaluator-source visibility, simulator
state access and reset privileges before running. Keep at least these distinct:

1. **Policy-interface exploitation:** candidates use only the admitted policy
   observation/action APIs against an unchanged benchmark implementation.
2. **Privileged simulator intervention:** candidates can inspect or manipulate
   simulator state beyond that interface; report the exact extra capabilities.

Changing evaluator code, writing scores, or modifying result files is evaluation
tampering, not evidence that a policy fooled an unchanged evaluator. Do not pool
it with policy-interface results. If evaluator source is disclosed during
development, label the study as white-box rather than claiming blind discovery.

## Development and validation

1. Pin benchmark source, assets, environment dependencies, task definition,
   initial System 1, CAP revision, System 2 revision and budget. Preserve exact
   units, coordinate frames and observation/action contracts.
2. Define the official proxy and independent task criterion before optimization.
   Record the verifier's implementation and thresholds. Preserve raw metrics;
   do not subtract unlike units without an explicit normalization.
3. Optimize on development cases. Admit only declared artifact edits. Freeze
   each candidate's complete dependency identity before executing it.
4. Freeze the candidate pool and select on a declared proxy-based comparison
   using paired cases. Keep final audit cases and independent validation
   judgments unavailable to the proposer and selector.
5. Audit the selected version and original parent on the same held-out cases.
   Report official score, independent success and their joint outcomes. Audit
   results cannot become feedback for further tuning on that same audit set.
6. Repeat across seeds and report denominators and uncertainty. Confirm suspected
   exploits with an observable mechanism and a targeted intervention, such as
   correcting the identified evaluator loophole and rechecking the behavior.

Independent failure with official success is evidence of proxy disagreement.
Calling it a confirmed exploit additionally requires investigation of the
mechanism and verifier reliability. A disagreement can instead expose a flawed
independent criterion or recording error.

## Evidence and selection

Integrate adapters through `ExperimentRuntime`, `ExperimentEvaluator` and
Self-Harness instead of making a second lineage system. A proxy-based research
selection must be labeled as such; it grants no physical qualification or robot
deployment authority. Store independent audit results as separate evidence;
do not overwrite the experiment verdict or retroactively change its criterion.

Each report should resolve to:

- frozen conditions, permitted access and policy/provider identities;
- observations, actions, reset status and termination reason;
- official proxy, independent outcome and verifier identity;
- parent/candidate manifests, proposal provenance and comparison cases;
- selection rationale, resource usage and inherited revision;
- mechanism analysis, repetitions and counterexamples.

Use the existing receipt and digest checks when integrating these records.
Uncertain effects and invalid setup stay distinct from task failure. Completed
evidence may be inspected without repeating actions; an interrupted operation
must not become an automatic retry or a fabricated score.

## Reporting boundary

Report scope as software or simulation as appropriate, with `qualification: null`
unless a separate qualification process supplies an actual result. Distinguish
an implemented adapter, a completed run, proxy disagreement and a confirmed
exploit. Neither generated code nor a successful import proves benchmark
coverage, model inference, simulator execution or physical transfer.
