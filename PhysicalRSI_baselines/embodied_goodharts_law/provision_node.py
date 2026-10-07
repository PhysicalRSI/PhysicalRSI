"""Install simulator dependencies on an assigned DSW's local disk.

This trusted deployment utility installs no candidate policy and reports no
benchmark score. Run it through the account-owned PAI execution transport.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['libero', 'libero-plus', 'robotwin'], required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--shared-root', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, help='Optional verified local wheel bundle')
    args = parser.parse_args()
    if os.environ.get('DSW_INSTANCE_ID') != args.instance:
        raise ValueError('Unexpected DSW instance')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    if uuids != [args.gpu_uuid]:
        raise ValueError('Assigned single GPU does not match the current instance')
    root, shared = args.local_root.resolve(), args.shared_root.resolve()
    if root.is_relative_to(shared) or shared.is_relative_to(root):
        raise ValueError('Use separate local and shared roots')
    root.mkdir(parents=True, exist_ok=True)
    with (root / ('provision-' + args.family + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = root / 'envs' / args.family
        constraint_family = 'robotwin' if args.family == 'robotwin' else 'libero'
        constraints = shared / 'config' / (constraint_family + '-native-constraints.txt')
        wheelhouse = None
        wheelhouse_sha256 = None
        if args.wheelhouse is not None:
            wheelhouse = args.wheelhouse.resolve()
            manifest = wheelhouse / 'manifest.json'
            contents = manifest.read_bytes()
            data = json.loads(contents)
            if data.get('state') != 'verified' or not data.get('wheels'):
                raise ValueError('Wheel bundle has not been verified')
            for row in data['wheels']:
                name = row['file']
                if Path(name).name != name or not name.endswith('.whl'):
                    raise ValueError('Wheel bundle file must be a wheel basename')
                path = wheelhouse / name
                if path.is_symlink() or path.stat().st_size != row['bytes']:
                    raise ValueError('Wheel bundle file differs from its manifest')
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                        raise ValueError('Wheel bundle digest mismatch')
            if set(p.name for p in wheelhouse.iterdir()) != {row['file'] for row in data['wheels']} | {'manifest.json'}:
                raise ValueError('Wheel bundle contains unlisted files')
            wheelhouse_sha256 = hashlib.sha256(contents).hexdigest()
        record = dict(schema='physicalrsi.egl-provision/v1', instance=args.instance,
                      gpu_uuid=args.gpu_uuid, family=args.family, state='installing',
                      started_at=time.time(), scope='simulator-dependency-install', qualification=None,
                      constraints_sha256=hashlib.sha256(constraints.read_bytes()).hexdigest(),
                      implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      wheelhouse_sha256=wheelhouse_sha256,
                      environment=str(target), benchmark_trial=False)
        receipt = shared / 'preflight' / 'nodes' / (args.instance + '.json')
        publish(receipt, record)
        (root / 'tmp').mkdir(exist_ok=True)
        env = dict(os.environ, UV_CACHE_DIR=str(root / 'cache/uv'), TMPDIR=str(root / 'tmp'),
                   XDG_CACHE_HOME=str(root / 'cache'), UV_LOCK_TIMEOUT='1800')
        uv = str(shared / 'bootstrap/bin/uv')
        try:
            with (root / ('provision-' + args.family + '.log')).open('ab') as log:
                if not (target / 'bin/python').exists():
                    python = '/usr/bin/python3.10' if args.family == 'robotwin' else '/usr/local/bin/python3.12'
                    command = [uv, 'venv', '--python', python]
                    if args.family != 'robotwin':
                        command.append('--system-site-packages')
                    subprocess.run(command + [str(target)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                command = [uv, 'pip', 'install', '--python', str(target / 'bin/python'), '-r', str(constraints)]
                if wheelhouse is not None:
                    command += ['--find-links', str(wheelhouse)]
                if args.family == 'libero-plus':
                    command += ['usd-core==25.5', 'wand==0.6.13', 'scikit-image==0.25.2']
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if args.family == 'robotwin':
                driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                  '--format=csv,noheader'], text=True).strip()
                if not re.fullmatch(r'[0-9.]+', driver):
                    raise ValueError('Unexpected driver version format')
                library = Path('/usr/lib/x86_64-linux-gnu') / ('libEGL_nvidia.so.' + driver)
                if not library.is_file():
                    raise ValueError('NVIDIA EGL library is missing')
                icd = root / 'nvidia_egl_icd.json'
                publish(icd, dict(file_format_version='1.0.0', ICD=dict(library_path=str(library), api_version='1.3.0')))
                record['vulkan_icd'] = str(icd)
            record.update(state='installed', finished_at=time.time())
        except Exception as error:
            record.update(state='failed', error_type=type(error).__name__, finished_at=time.time())
            publish(receipt, record)
            raise
        publish(receipt, record)
        print(json.dumps(record))


if __name__ == '__main__':
    main()
