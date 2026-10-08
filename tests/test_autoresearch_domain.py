"""Domain extension must preserve the upstream research engine's evidence gates."""
from pathlib import Path
import pytest
from PhysicalRSI_Autoresearch.study import run_study
from PhysicalRSI_core.infra.storage import file_digest


class Domain:
    identity = {'kind': 'test-only-domain'}

    def __init__(self):
        self.calls = []

    def proposals_checked(self, proposals):
        return [dict(name=k, hypothesis='fixture', settings={}) for k in ['baseline', 'better', 'tie']]

    def sources(self):
        return {}

    def manifest(self):
        return dict(scope='test-only', question='fixture', metrics=['failures', 'final_failures'],
                    development=[{'seed': 0}], validation=[{'seed': 1}], selection='Minimize failures')

    def evaluate(self, proposal, spec, split, output, timeout):
        self.calls.append((proposal['name'], split))
        (output/'trace.txt').write_text(proposal['name'])
        score = 2 if proposal['name'] == 'baseline' else 1
        return dict(audit=dict(failures=score, final_failures=score), simulation=dict(passed=True),
                    passed=split == 'development')


def test_domain_resume_ties_and_validation_do_not_change_selection(tmp_path):
    d = Domain(); root = tmp_path/'study'
    run_study(root, d, domain=d, max_rounds=1)
    result = run_study(root, d, domain=d, resume=True)
    assert result['selected'] == 'better'
    assert result['validation_passed'] is False
    assert [r['status'] for r in result['rounds']] == ['keep', 'keep', 'discard']
    assert d.calls == [('baseline','development'), ('better','development'), ('tie','development'), ('better','validation')]
    assert 'final_failures' in (root/'results.tsv').read_text()
    run_study(root, d, domain=d, resume=True)
    assert len(d.calls) == 4
    receipt = root/result['evidence'][0]['path']
    (receipt.parent/'0/trace.txt').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        run_study(root, d, domain=d, resume=True)


@pytest.mark.parametrize('kind', ['absolute', 'parent', 'symlink'])
def test_source_snapshot_rejects_paths_outside_repository_before_writing(tmp_path, monkeypatch, kind):
    import os
    import PhysicalRSI_Autoresearch.study as engine
    repo = tmp_path / 'repository'
    repo.mkdir()
    outside = tmp_path / 'outside.py'
    outside.write_text('outside fixture must remain untouched\n')
    os.utime(outside, ns=(1_000_000, 1_000_000))
    before = outside.stat().st_mtime_ns
    monkeypatch.setattr(engine, '__file__', str(repo / 'engine/study.py'))
    monkeypatch.setattr(engine, '_sources', lambda: {})
    if kind == 'absolute':
        key = str(outside)
    elif kind == 'parent':
        key = '../outside.py'
    else:
        (repo / 'linked.py').symlink_to(outside)
        key = 'linked.py'
    domain = Domain()
    domain.sources = lambda: {key: file_digest(outside)}
    root = tmp_path / 'study'
    with pytest.raises(ValueError, match='Source path'):
        run_study(root, domain, domain=domain)
    assert not root.exists()
    assert outside.read_text() == 'outside fixture must remain untouched\n'
    assert outside.stat().st_mtime_ns == before


def test_resume_rejects_source_snapshot_symlink_escape(tmp_path, monkeypatch):
    import PhysicalRSI_Autoresearch.study as engine
    repo = tmp_path / 'repository'
    repo.mkdir()
    source = repo / 'source.py'
    source.write_text('fixture\n')
    outside = tmp_path / 'outside.py'
    outside.write_bytes(source.read_bytes())
    monkeypatch.setattr(engine, '__file__', str(repo / 'engine/study.py'))
    monkeypatch.setattr(engine, '_sources', lambda: {})
    domain = Domain()
    domain.sources = lambda: {'source.py': file_digest(source)}
    root = tmp_path / 'study'
    run_study(root, domain, domain=domain)
    snapshot = root / 'sources/source.py'
    snapshot.unlink()
    snapshot.symlink_to(outside)
    with pytest.raises(ValueError, match='Source snapshot'):
        run_study(root, domain, domain=domain, resume=True)
