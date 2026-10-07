"""Capture installed distribution bytes for native-provider admission.

Run with the provider interpreter. This inventory does not attest unregistered
source trees, simulator assets, driver libraries, environment variables or a
complete execution closure. It records discrepancies rather than blessing them.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import sysconfig


def file_identity(path):
    before = path.stat()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                              after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('File changed while hashing: ' + str(path))
    return {'path': str(path.absolute()), 'resolved_path': str(path.resolve()),
            'bytes': after.st_size, 'sha256': sha.hexdigest()}


def capture():
    distributions = []
    cache = {}
    issues = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name', '<unnamed>')
        row = {'name': name, 'version': distribution.version, 'files': []}
        # Some Python versions filter missing entries from Distribution.files.
        # Parse RECORD directly so removed installed files remain discrepancies.
        record = distribution.read_text('RECORD')
        if record is not None:
            files = []
            for filename, recorded_hash, size in csv.reader(io.StringIO(record)):
                entry = importlib.metadata.PackagePath(filename)
                entry.hash = importlib.metadata.FileHash(recorded_hash) if recorded_hash else None
                entry.size = int(size) if size else None
                files.append(entry)
        else:
            files = distribution.files
            issues.append({'distribution': name, 'problem': 'no-RECORD-missing-file-check-incomplete'})
        if files is None:
            issues.append({'distribution': name, 'problem': 'no-installed-file-list'})
            distributions.append(row)
            continue
        for entry in sorted(files, key=str):
            # Bytecode caches are regenerated from the separately captured source.
            if str(entry).endswith(('.pyc', '.pyo')):
                continue
            path = Path(distribution.locate_file(entry)).absolute()
            try:
                if str(path) not in cache:
                    cache[str(path)] = file_identity(path)
                identity = dict(cache[str(path)], distribution_path=str(entry))
            except (OSError, RuntimeError) as exc:
                issues.append({'distribution': name, 'path': str(path),
                               'problem': type(exc).__name__, 'detail': str(exc)})
                continue
            if entry.hash is not None:
                identity['record_hash'] = {'mode': entry.hash.mode, 'value': entry.hash.value}
                if entry.hash.mode == 'sha256':
                    actual = base64.urlsafe_b64encode(bytes.fromhex(identity['sha256'])).rstrip(b'=').decode()
                    if actual != entry.hash.value:
                        issues.append({'distribution': name, 'path': str(path), 'problem': 'RECORD-hash-mismatch'})
                else:
                    issues.append({'distribution': name, 'path': str(path), 'problem': 'unsupported-RECORD-hash'})
            if entry.size is not None and entry.size != identity['bytes']:
                issues.append({'distribution': name, 'path': str(path), 'problem': 'RECORD-size-mismatch'})
            row['files'].append(identity)
        distributions.append(row)
    # Editable installs may reference code outside installed distribution records.
    editable = []
    for row in distributions:
        for entry in row['files']:
            if entry['distribution_path'].endswith('direct_url.json'):
                data = json.loads(Path(entry['path']).read_text())
                if data.get('dir_info', {}).get('editable'):
                    editable.append({'distribution': row['name'], 'source_url': data.get('url')})
    return {'schema': 'physicalrsi.egl-environment-inventory/v1',
            'scope': 'installed-distribution-byte-inventory', 'qualification': None,
            'complete_execution_closure_attested': False,
            'inventory_implementation': file_identity(Path(__file__)),
            'interpreter_invocation': sys.executable,
            'interpreter': file_identity(Path(sys.executable)),
            'python_version': sys.version, 'prefix': sys.prefix,
            'base_prefix': sys.base_prefix, 'sys_path': sys.path,
            'platform': sysconfig.get_platform(),
            'distributions': sorted(distributions, key=lambda x: (x['name'].lower(), x['version'])),
            'unique_files': len(cache), 'unique_file_bytes': sum(x['bytes'] for x in cache.values()),
            'editable_sources_requiring_separate_inventory': editable,
            'issues': issues,
            'limits': ['Capture is not an atomic filesystem snapshot; freeze providers and reverify before admission',
                       'Does not capture stdlib, external shared libraries, source trees, model assets or runtime configuration',
                       'Installed bytes are identified even when RECORD differs; discrepancies require review']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a fresh inventory output')
    result = capture()
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'distributions': len(result['distributions']),
                      'unique_files': result['unique_files'], 'issues': len(result['issues']),
                      'editable_sources': len(result['editable_sources_requiring_separate_inventory'])}))


if __name__ == '__main__':
    main()
