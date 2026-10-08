from pathlib import Path

import pytest

from PhysicalRSI_baselines.robotworld.asset_admission import verify_assets
from PhysicalRSI_core.infra.storage import atomic_json, file_digest


@pytest.fixture
def release(tmp_path):
    root = tmp_path / 'Assets'
    rows = []
    for name in ('robocasa/data/objects/cup/model.xml', '_shared/texture.png'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture asset')
        rows.append(dict(path='Assets/' + name, bytes=path.stat().st_size, sha256=file_digest(path)))
    manifest = tmp_path / 'manifest.json'
    atomic_json(manifest, {'files': rows})
    return root, manifest, file_digest(manifest)


def test_complete_population_and_intentional_root_link(release, tmp_path):
    root, manifest, sha = release
    link = tmp_path / 'asset-mount'
    link.symlink_to(root, target_is_directory=True)
    report = verify_assets(link, manifest, manifest_sha256=sha)
    assert report['files'] == 2
    assert report['qualification'] is None


@pytest.mark.parametrize('change', ['missing', 'extra', 'corrupt', 'symlink'])
def test_changed_sampling_population_or_content_is_rejected(release, change):
    root, manifest, sha = release
    path = root / 'robocasa/data/objects/cup/model.xml'
    if change == 'missing':
        path.unlink()
    elif change == 'extra':
        (path.parent / 'another.xml').write_text('extra')
    elif change == 'corrupt':
        path.write_bytes(b'Fixture asset')
    else:
        path.unlink()
        path.symlink_to(root / '_shared/texture.png')
    with pytest.raises(ValueError):
        verify_assets(root, manifest, manifest_sha256=sha)


def test_replaced_manifest_cannot_redefine_expected_population(release):
    root, manifest, sha = release
    atomic_json(manifest, {'files': []})
    with pytest.raises(ValueError, match='manifest changed'):
        verify_assets(root, manifest, manifest_sha256=sha)


def test_manifest_without_object_population_is_rejected(tmp_path):
    manifest = tmp_path / 'manifest.json'
    atomic_json(manifest, {'files': []})
    with pytest.raises(ValueError, match='sampling population'):
        verify_assets(tmp_path, manifest, manifest_sha256=file_digest(manifest))
