# Native RoboDyna adapter

This adapter runs the official RoboDyna tasks in an isolated Python process and
connects them to PhysicalRSI's existing experiment and Self-Harness interfaces.
It supports `catch_ramp_ball` and `catch_valley_ball`. Both use the same
`dynamic_interception` composition; grasping the cup and pushing the box belong
to the execution adapter. Native task code supplies success and progress.

## Runtime separation

Use Python 3.11 or newer for PhysicalRSI and a separate Python 3.10 environment
for RoboDyna. Do not install the simulator dependency stack into the framework's
environment. Keep upstream checkouts, downloaded assets, build caches and worker
outputs outside this repository.

The environment used for the native reset and complete rollout checks was:

| Component | Executed version |
|---|---|
| RoboDyna | `6422e622d8025c3eaa4559bb4250c968fe4b2873` |
| CuRobo | v0.7.8, commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| Native interpreter | Python 3.10 |
| PyTorch | `2.7.1+cu128` |
| torchvision | `0.22.1+cu128` |
| SAPIEN | `3.0.3` |
| NumPy | `1.26.4` |
| Open3D | `0.20.0` |
| CUDA extension compiler | CUDA 12.1, `TORCH_CUDA_ARCH_LIST=8.9+PTX` |
| GPU and driver | NVIDIA RTX PRO 5000 72GB Blackwell, driver 580.126.09 |

CuRobo's compiled extensions were exercised through native reset and a complete
robot rollout on this GPU, including the forward-JIT path. This describes an
executed configuration, not a guarantee for every driver or GPU. The retained
[requirements-native.txt](requirements-native.txt) contains the pinned installed
package inventory, with CuRobo installed separately from its pinned source.
Runtime records retain the dependency inventory, source revisions, hardware,
build logs and native rollout evidence.

## Prepare the isolated dependencies

Choose local directories and export them before running these examples:

```bash
export ROBODYNA_ROOT=/path/to/isolated/robodyna
export CUROBO_ROOT=/path/to/isolated/curobo
export ROBODYNA_VENV=/path/to/isolated/python310-environment
export ROBODYNA_RUNTIME=/path/to/isolated/runtime-records
export ROBODYNA_OUTPUT=/path/to/isolated/worker-output
```

Clone [RoboDyna](https://github.com/robodyna/robodyna) and
[CuRobo](https://github.com/NVlabs/curobo) into those directories and check out
the exact commits above. Create the native environment with Python 3.10. Set
`PHYSICALRSI_ROOT` to this repository's checkout. Install the pinned inventory, then build CuRobo without resolving its dependencies again:

```bash
"$ROBODYNA_VENV/bin/python" -m pip install -r "$PHYSICALRSI_ROOT/PhysicalRSI_baselines/robodyna/requirements-native.txt"
export CUDA_HOME=/path/to/cuda-12.1
export TORCH_CUDA_ARCH_LIST='8.9+PTX'
"$ROBODYNA_VENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$CUROBO_ROOT"
```

The build needs the matching Python development headers and CUDA compiler, in
addition to the NVIDIA driver. The PyTorch wheel's bundled CUDA runtime and the
extension compiler are separate installations. Preserve the install log and
export a fresh dependency inventory after a successful reconstruction.

Download assets with the upstream downloader, using the native environment:

```bash
export PATH="$ROBODYNA_VENV/bin:$PATH"
cd "$ROBODYNA_ROOT"
bash script/_download_assets.sh
```

The downloader checks the archive against its published SHA-256 checksum. It
also invokes `script/update_embodiment_config_path.py` to expand robot asset
paths for this machine. Those generated path configurations are expected to
change; task source and native scoring remain upstream-owned. The adapter hashes
runtime task/configuration files into its experiment identity.

## Headless GPU setup

First inspect the selected GPU with `nvidia-smi`, including driver, memory and
utilization. Start with one worker. CUDA visibility does not prove which Vulkan
device SAPIEN selected; verify renderer initialization and actual GPU usage with
a native reset before expanding concurrency.

Use the container's NVIDIA Vulkan and EGL manifests when present. If the driver
libraries are mounted but their manifests are absent, create local manifests:

```bash
mkdir -p "$ROBODYNA_RUNTIME"
cat > "$ROBODYNA_RUNTIME/nvidia_icd.json" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}
JSON
cat > "$ROBODYNA_RUNTIME/nvidia_egl.json" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
export VK_ICD_FILENAMES="$ROBODYNA_RUNTIME/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$ROBODYNA_RUNTIME/nvidia_egl.json"
export CUDA_VISIBLE_DEVICES=0
unset DISPLAY
```

Freeze these explicit settings into the study identity rather than relying only
on inherited shell variables:

```bash
python - <<'PY'
import json, os
from pathlib import Path
keys = ('VK_ICD_FILENAMES', '__EGL_VENDOR_LIBRARY_FILENAMES', 'CUDA_VISIBLE_DEVICES')
path = Path(os.environ['ROBODYNA_RUNTIME']) / 'environment.json'
path.write_text(json.dumps({key: os.environ[key] for key in keys}, indent=2))
PY
```

Pass `--runtime-env "$ROBODYNA_RUNTIME/environment.json"` to the native study.
Preserve the referenced manifest files alongside the runtime configuration.

These files select existing NVIDIA libraries; they do not install a driver.
The adapter disables ray tracing by default and removes `DISPLAY` from the worker
environment. Each episode receives its own process, log and evidence directory.

## Native lifecycle check

Run the following from the PhysicalRSI checkout with its framework interpreter.
It exercises the real task's reset, simulation step, native scoring and evidence
finalization. It does not estimate benchmark performance.

```python
import os
from PhysicalRSI_baselines.robodyna.adapter import NativeSuite, NativeEnvironment
from PhysicalRSI_core.contracts import Context

suite = NativeSuite(
    os.environ['ROBODYNA_ROOT'],
    os.environ['ROBODYNA_VENV'] + '/bin/python',
    output_root=os.environ['ROBODYNA_OUTPUT'],
    runtime_environment={key: os.environ[key] for key in (
        'VK_ICD_FILENAMES', '__EGL_VENDOR_LIBRARY_FILENAMES',
        'CUDA_VISIBLE_DEVICES')},
)
environment = NativeEnvironment(suite, 'catch_ramp_ball')
context = Context('native-lifecycle-check')
try:
    environment.reset({'seed': 0, 'scenario': 'default', 'mode': 'frozen'}, context)
    environment.step({'kind': 'observe', 'steps': 100}, context)
    print(environment.native_success(), environment.native_progress())
finally:
    environment.close()
print(environment.episode_evidence())
```

## Observation and evaluation semantics

The integration mode is recorded as `native-interactive-robot`, with
`simulator-state-and-static-geometry` observations. The worker enables upstream's
`_interactive_robot_mode` before reset. This is necessary for contact-driven box
physics in Valley Ball; a frozen scene-interactive box is not an equivalent robot
execution mode. Results belong to this declared mode and must not be presented as
runs of the published visual policy evaluator.

The shared skill receives timestamped ball positions, a motion phase, static
surface geometry and a catcher reachability envelope. These are privileged
simulator observations, not visual estimates. Predetermined landing coordinates,
future launch velocity and baked future ball lanes are excluded. Observation
history resets at phase transitions. The skill predicts a capture point; native
execution remains responsible for grasping, pushing and robot motion.

`frozen` does not add inference latency to simulated motion. `async` enables the
upstream latency mechanism; the worker records the inference timing scope.
Report the selected mode explicitly. Scenarios are configured through upstream's
`configure_evaluation`; final evaluation seeds must remain separate from
candidate development and selection. Retain worker logs, trajectory/event
records, rendered video and source identities alongside campaign evidence.
Report task success and progress only from completed native evaluations, with
their counts, seeds, latency and uncertainty.
