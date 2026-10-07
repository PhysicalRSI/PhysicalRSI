import sys
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.renderer_identity import current_renderer_identity


@pytest.mark.parametrize("vendor", [None, b"Mesa", b""])
def test_missing_or_software_context_cannot_claim_nvidia(monkeypatch, vendor):
    gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, GL_VERSION=3,
                         glGetString=lambda key: {1: vendor, 2: b"llvmpipe", 3: b"4.5"}[key])
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
    with pytest.raises(RuntimeError):
        current_renderer_identity(require_nvidia=True)


def test_actual_context_identity_is_retained(monkeypatch):
    gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, GL_VERSION=3,
                         glGetString=lambda key: {1: b"NVIDIA Corporation", 2: b"Test device", 3: b"4.6"}[key])
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
    result = current_renderer_identity(require_nvidia=True)
    assert result["renderer"] == "Test device"
    assert result["version"] == "4.6"
    assert len(result["implementation_sha256"]) == 64
