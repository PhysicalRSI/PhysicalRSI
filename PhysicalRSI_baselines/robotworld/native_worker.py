"""Trusted candidate-loading host worker for RoboCasaTrials; no score claims."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

from PhysicalRSI_core.infra.storage import atomic_json, digest, file_digest, read_json
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_baselines.robotworld import asset_admission, isolation, policy_bundle
from PhysicalRSI_baselines.robotworld.native_trials import validate_case
from PhysicalRSI_baselines.robotworld.receipts import robocasa_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('deployment', 'request', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    deployment, request = read_json(args.deployment), read_json(args.request)
    if request.get('schema') != 'physicalrsi.robotworld-native-trial/v1':
        raise ValueError('Expected a frozen native trial request')
    case, candidate = request['case'], request['candidate']
    validate_case(case)
    sha = verify_harness(candidate)
    if sha != request['candidate_sha256']:
        raise ValueError('Candidate differs from the frozen request')
    upstream = Path(deployment['upstream']).resolve(strict=True)
    episode = Path(__file__).with_name('robocasa_episode.py')
    asset_source = upstream / 'environment/datasets/asset-source.json'
    sources = deployment['sources']
    required = {str(p.resolve()) for p in (upstream / 'environment').rglob('*.py')}
    required.update(str(p.resolve()) for p in (
        Path(__file__), episode, Path(asset_admission.__file__), Path(isolation.__file__),
        Path(policy_bundle.__file__), Path(deployment['native_python']),
        Path(deployment['build_manifest']), Path(deployment['bubblewrap']), asset_source))
    for directory in deployment['native_pythonpath']:
        if Path(directory).resolve() != upstream:
            required.update(str(p.resolve()) for p in Path(directory).rglob('*.py'))
    if not required <= set(sources):
        raise ValueError('Deployment omits native executable/source inputs')

    def stable():
        for name, expected in sources.items():
            if file_digest(Path(name)) != expected:
                raise ValueError('Deployment source changed')

    stable()
    if read_json(asset_source)['manifest_sha256'] != deployment['asset_manifest_sha256']:
        raise ValueError('Asset manifest differs from the upstream release')
    asset_report = asset_admission.verify_assets(deployment['asset_root'], deployment['asset_manifest'],
                                                manifest_sha256=deployment['asset_manifest_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    policy_root = args.output / 'policy-inputs'
    bundle = policy_bundle.materialize(candidate, fixed=deployment['fixed_components'], destination=policy_root)
    atomic_json(args.output / 'policy-inputs.json', bundle)
    atomic_json(args.output / 'asset-admission.json', asset_report)
    home = Path(deployment['codex_home'])
    config = tomllib.loads((home / 'config.toml').read_text())
    if (config.get('model') != deployment['model']
            or config.get('model_provider', 'openai') != deployment['provider']
            or config.get('model_reasoning_effort') != deployment['reasoning_effort']):
        raise ValueError('Model/provider configuration differs from the frozen deployment')
    sys.path.insert(0, str(upstream))
    from environment.runtime import isolated_codex
    os.environ['PATH'] = str(Path(deployment['bubblewrap']).parent) + os.pathsep + os.environ['PATH']
    os.environ['WORLD_CODEX_DISABLE_CODE_MODE'] = '1'
    os.environ['WORLD_AGENT_BACKEND'] = 'bubblewrap'
    os.environ['TMPDIR'] = deployment['runtime_tmp']
    tempfile.tempdir = deployment['runtime_tmp']
    original = isolated_codex.sandbox_command

    def sandbox(*positional, **kwargs):
        command = isolation.mount_policy(isolation.minimal_proc_command(original(*positional, **kwargs)), policy_root)
        settings = []
        for name in ('http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY'):
            if name in os.environ:
                settings += ['--setenv', name, os.environ[name]]
        index = command.index('--clearenv') + 1
        command[index:index] = settings
        return command

    isolated_codex.sandbox_command = sandbox
    with isolated_codex.IsolatedCodex(deployment['build_manifest'], args.output, home) as relay:
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(deployment['native_pythonpath']),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', NUMBA_CACHE_DIR=deployment['numba_cache'],
                   WORLD_CODEX_SOCKET=str(relay.socket_path), WORLD_AGENT_OBSERVATIONS=str(relay.observations))
        # Inherit the runner's process group: its bounded teardown owns simulator descendants.
        command = [deployment['native_python'], str(episode), '--request', str(args.request),
                   '--output', str(args.output), '--manifest', deployment['build_manifest'],
                   '--policy-inputs', str(policy_root)]
        atomic_json(args.output / 'native-command.json', {'command': command})
        with (args.output / 'native.log').open('w') as log:
            subprocess.run(command, cwd=upstream, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    stable()
    if verify_harness(candidate) != sha or any(file_digest(policy_root / name) != value
                                             for name, value in bundle['files'].items()):
        raise ValueError('Frozen candidate or mounted inputs changed during execution')
    robocasa_receipt(args.output, task=case['task'], seed=case['reset_seed'], horizon=case['horizon'])
    atomic_json(args.output / 'execution-binding.json', {'request_sha256': digest(request)})


if __name__ == '__main__':
    main()
