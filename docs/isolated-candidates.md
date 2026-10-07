# Isolated System 1 and System 2 programs

Candidate Python can run in a separate Linux process with a restricted filesystem, syscall filter, nonprivileged UID and bounded resources. The trusted host supplies only the declared input data. The candidate has no network sockets, robot SDK, TLS keys, host workspace or verifier handle. The host receives JSON data and remains responsible for action admission and independent scoring.

This adapts a useful separation in the [reviewed OpenMLE Sandbox](https://github.com/FrontisAI/OpenRSI/blob/71ae803a035d5e3b78c19fa49ed9f67d0550cbaa/OpenMLE-Gym/openmle-sandbox/README.md): scheduling/executing a job and evaluating its result are separate responsibilities. That upstream snapshot also explicitly describes its private task directory as a convention rather than an enforced access boundary. Here, only deliberately supplied source, inputs and a pinned Python runtime enter the candidate's filesystem. This is an independent extension of physicalRSI's existing chroot/seccomp executor.

## Two distinct interfaces

| Adapter | Candidate interface | Lifetime | Host responsibility |
| --- | --- | --- | --- |
| `IsolatedPolicy` | `begin_episode(task, case, observation)` returns transient state; `act(observation, state)` returns exactly `{"action": ..., "state": ...}` | One process per episode, retaining in-process state | Dispatch returned actions through the existing experiment/controller boundary; verify the resulting observations |
| `IsolatedProposal` | `generate(request, limits)` returns the existing structured proposal reply | One process per proposal call | Validate edits, materialize candidates, run independent trials, select and record lineage |

System 1 can use its observations, public case and explicitly supplied memory throughout the episode. Its returned state is transient; inheritable memory still needs an immutable revision. System 2 receives the existing development-only proposal projection and edit schema. It receives no primitive callbacks. A model provider's JSON reply still goes through the existing schema admission; the networked `ModelProposal` adapter remains a trusted transport and is not automatically moved into this network-free profile.

The candidate may replace or bypass the helper code inside its own process. The security boundary is the OS setup plus the host's limited data protocol, not Python object hiding or a restricted `exec` namespace. Only the host may turn a proposed action into a device call. Ordinary controller validation, current-generation claims, observation freshness and watchdog behavior continue to apply.

## Runtime and trust boundary

`PythonIsolation` requires Linux, chroot privilege and libseccomp. It fails if unavailable; there is no unrestricted fallback. A trusted operator builds a stdlib-only runtime with `infra.isolation.build_runtime()`. The runtime manifest pins its executable and copied library files. Each launch verifies the manifest and files before staging a read-only jail using hard links. Runtime and execution directories must be on the same filesystem and protected from untrusted host processes.

The existing launcher drops to UID/GID 65534, clears supplementary groups and the environment, closes extra descriptors, sets no-new-privileges and parent-death SIGKILL, then installs a syscall allowlist before executing candidate Python. No `/proc` or device filesystem is supplied. Socket creation, fork/clone, signaling other processes and privilege changes are outside the allowlist. CPU, address-space, open-file and file-size limits bound the child. Only its precreated result mailbox is writable inside the jail; stdout/stderr go to its owned bounded log.

[Linux's seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html) explains the syscall filtering mechanism and why a filter alone is not a complete sandbox. This profile combines filtering with filesystem and credential restrictions. It shares the host kernel and does not claim resistance to kernel vulnerabilities, isolation from a privileged host operator, GPU support or hard real-time behavior. The runtime, launcher, host callbacks and kernel are trusted. Pinned files and successful boundary probes are evidence for this profile, not a universal security certification.

Preparation and execution consume the same absolute episode/proposal deadline. File hashing and filesystem I/O remain host operations and cannot interrupt an indefinitely stalled filesystem syscall. Normal cancellation and deadline expiry trigger bounded owned-process teardown. A cleanup timeout remains unresolved and cannot release a device claim. No extra worker threads or subprocesses are available to the candidate.

## Integrate an isolated policy

Use an operator-built runtime on the trusted host:

```python
from PhysicalRSI_core.embodiment import System1
from PhysicalRSI_core.infra.isolated_policy import IsolatedPolicy
from PhysicalRSI_core.infra.isolated_program import PythonIsolation
from PhysicalRSI_core.infra.storage import digest
from PhysicalRSI_demos.experiment import OBSERVATION, ACTION

source = """
def begin_episode(task, case, observation):
    return {"steps": 0}

def act(observation, state):
    return {"action": 1, "state": {"steps": state["steps"] + 1}}
"""
profile = PythonIsolation("/tmp/operator-prepared-python-runtime")
policy = IsolatedPolicy(
    source,
    specification=System1("isolated-counter", digest(source), OBSERVATION, ACTION),
    isolation=profile,
    output="/tmp/physicalrsi-policy-workers",
    max_actions=100,
    call_seconds=2,
)
```

Pass `policy` to the normal `ExperimentRuntime`. For System 2 trials, declare the executing candidate's verified harness revision as the System 1 revision. The adapter identity additionally pins source, isolation configuration and its own implementation. Returning an action packet does not exempt it from the environment's contract and controller checks. Use unique execution directories for separately allocated policy instances; an instance rejects reuse of an existing episode directory.

`ExperimentRuntime` supports the optional `end_episode(context)` policy method. It runs after action production and on failed trials, including failed initialization. Implementations must confirm `{"stopped": true}` after their own bounded cleanup even when the context has expired. A normal completed trial includes the hashed `policy.json` cleanup report. `IsolatedPolicy` binds that report to its episode and rejects cleanup requests for a different episode. Unconfirmed cleanup leaves the trial and any device lease unresolved. Policy termination and device quiescence are recorded separately; stopping an inference worker does not assert that a robot stopped.

The persistent bridge accepts only ordered data exchanges. It rejects unexpected methods, changed call IDs, duplicate JSON members, non-finite values, oversized messages and late responses. The trusted host checks call and episode budgets. Killing the experiment owner kills its isolated child through the kernel parent-death mechanism; an interrupted experiment still requires reconciliation.

`IsolatedProposal(source, isolation=profile, output=...)` implements the ordinary `identity()` / `generate()` strategy port and can be supplied to `StructuredProposer`. It cannot invoke the policy exchange port or a controller method. The existing durable proposal call record prevents automatic retry of an unknown or timed-out call. Process exit and claimed proposal quality never become a success score.

## Bind executable skills to candidates

`self_harness.code_policy.IsolatedHarnessPolicy` loads a declared Python skill from a verified candidate closure. The skill file belongs exclusively to the `skills` component and contains:

```json
{
  "schema": "physicalrsi.python-skill/v1",
  "source": "def begin_episode(task, case, observation): return None\ndef act(observation, state): return dict(action=1, state=state)\n"
}
```

The caller supplies a `System1` specification whose revision equals `verify_harness(candidate)`. The adapter adds a `skill` dependency named after the relative artifact path with its exact file SHA256. A conflicting dependency, shared ownership by another component, malformed artifact or changed closure is rejected. Loading performs bounded JSON reads and digest checks; it never imports, compiles or executes candidate source in the host. The source enters the existing restricted worker when the experiment begins.

```python
from PhysicalRSI_core.self_harness.code_policy import (
    IsolatedHarnessPolicy, PYTHON_SKILL_SCHEMA,
)
from PhysicalRSI_core.self_harness.proposals import JsonEditContract

contract = JsonEditContract({
    "skills/controller.json": {"component": "skills", "schema": PYTHON_SKILL_SCHEMA},
})
policy = IsolatedHarnessPolicy(
    candidate,
    skill_file="skills/controller.json",
    specification=system1,  # Exact candidate revision and environment interfaces.
    isolation=profile,
    output=worker_directory,
)
```

Supply this edit contract to `StructuredProposer` and construct the policy in the suite's policy factory. Source changes then use the ordinary proposal journal, separate materialization, admission, paired trials and lineage. A strategy may be reviewed enumeration, an isolated generator or a configured model provider; the artifact format does not establish the quality of generated code. The suite must explicitly admit executable skill edits and retain independent action limits, device ownership, stop behavior and scoring. Schema validation is not a claim that a program is correct or appropriate for hardware. Dependencies besides the supplied stdlib runtime are not automatically installed or exposed.

`tests/test_harness_code_policy.py` demonstrates an actual source edit replacing a reviewed counter program, execution in restricted processes, measured inheritance and recovery without new effects. It also rejects wrong revision attribution and source drift, and checks that malicious top-level source is not executed during host-side loading. This is software integration evidence; it uses no model generation or physical device.

## Reproduce the evidence

The CPU tests build a temporary runtime and execute actual restricted subprocesses:

```bash
python -m pytest -q tests/test_isolated_policy.py tests/test_harness_code_policy.py tests/test_embodied_experiments.py
```

They test state persistence/reset, absent host credentials and ambient secrets, blocked network/device/process access, forged capabilities, malformed/oversized output, hangs, cancellation, process-owner death, runtime changes, durable cleanup evidence, retained device claims and isolated System 2 proposals evaluated by the regular campaign. Missing Linux privileges or libseccomp explicitly skips OS isolation tests; lifecycle tests remain available without them.

The optional MuJoCo campaign exercises the same interfaces across force and torque tasks:

```bash
PHYSICALRSI_RUNTIME="$(mktemp -d /tmp/physicalrsi-python-runtime.XXXXXX)/python"
python -c 'import sys; from PhysicalRSI_core.infra.isolation import build_runtime; build_runtime(sys.argv[1])' "$PHYSICALRSI_RUNTIME"
python -m PhysicalRSI_demos.mujoco_evolution \
  --workspace "$(mktemp -d /tmp/physicalrsi-isolated-campaign.XXXXXX)" \
  --isolation-runtime "$PHYSICALRSI_RUNTIME" --proposer isolated
```

Run with the optional MuJoCo environment and the required Linux privilege. `--isolation-runtime` moves the reviewed System 1 code into one restricted process per trial. `--proposer isolated` additionally moves the deterministic reviewed proposal program into its own restricted process. Both produce data; the existing independently scored trials and lineage decide inheritance. These switches do not generate novel code, train weights or add a robot dependency to core.

MuJoCo is a repeatable test environment here. Robot adapters still own their sensor mapping, unit/frame contracts, device SDK access, actuator limits, independently enforced stop behavior and physical qualification. This profile supplies no real robot or external model evidence. Reports retain simulation scope and `qualification: null`.
