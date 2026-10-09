"""Check source binding with separately installed Core and GPT-as-Policy."""
import pytest

from PhysicalRSI_demos.dexterous_manipulation import common
from PhysicalRSI_demos.dexterous_manipulation.dexjoco_campaign import freeze_candidate, verify_executing_source


def test_external_dependency_edit_or_added_source_invalidates_candidate(tmp_path, monkeypatch):
    roots = {}
    for name in ("adapter", "gpt_as_policy", "physicalrsi_core"):
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("# source fixture\n")
        roots[name] = root
    monkeypatch.setattr(common, "source_roots", lambda: roots)
    candidate = freeze_candidate(tmp_path / "candidate", "layout-fixture")
    verify_executing_source(candidate)
    dependency = roots["physicalrsi_core"] / "__init__.py"
    dependency.write_text("# changed dependency\n")
    with pytest.raises(ValueError, match="frozen closure"):
        verify_executing_source(candidate)
    dependency.write_text("# source fixture\n")
    added = roots["gpt_as_policy"] / "additional_module.py"
    added.write_text("# newly added source\n")
    with pytest.raises(ValueError, match="frozen closure"):
        verify_executing_source(candidate)
    added.unlink()
    verify_executing_source(candidate)
    dependency.unlink()
    with pytest.raises(ValueError, match="frozen closure"):
        verify_executing_source(candidate)
