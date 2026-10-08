"""Verify the complete pinned RoboCasa/shared asset population before reset."""
from pathlib import Path

from PhysicalRSI_core.infra.storage import file_digest, read_json


def verify_assets(root, manifest, *, manifest_sha256):
    root = Path(root).resolve(strict=True)
    manifest = Path(manifest)
    if file_digest(manifest) != manifest_sha256:
        raise ValueError('Asset manifest changed')
    expected = {}
    for row in read_json(manifest)['files']:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or len(path.parts) < 3 or path.parts[0] != 'Assets':
            raise ValueError('Unsafe asset manifest path')
        if path.parts[1] not in {'robocasa', '_shared'}:
            continue
        name = path.relative_to('Assets').as_posix()
        if name in expected or type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Invalid or duplicate asset entry')
        expected[name] = row
    if not expected or not any(name.startswith('robocasa/data/objects/') for name in expected):
        raise ValueError('Manifest omits the object sampling population')
    observed = set()
    for bench in ('robocasa', '_shared'):
        directory = root / bench
        if directory.is_symlink():
            raise ValueError('Asset subtree must contain physical files')
        for path in directory.rglob('*'):
            if path.is_symlink():
                raise ValueError('Asset subtree must contain physical files')
            if path.is_file():
                observed.add(path.relative_to(root).as_posix())
    if observed != set(expected):
        raise ValueError('Asset population mismatch: missing=%d unexpected=%d' % (
            len(set(expected) - observed), len(observed - set(expected))))
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']:
            raise ValueError('Asset content differs from the pinned release: ' + name)
    return dict(manifest_sha256=manifest_sha256, files=len(expected),
                bytes=sum(row['bytes'] for row in expected.values()), qualification=None)
