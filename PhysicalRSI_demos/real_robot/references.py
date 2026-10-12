"""Resolve the installed PhysicalRSI source; no sibling checkout is required."""
from pathlib import Path
import PhysicalRSI_core

RSI = Path(PhysicalRSI_core.__file__).resolve().parent.parent


def use_physicalrsi():
    """Compatibility hook; normal package imports select the implementation."""
    if not (RSI / "PhysicalRSI_core/__init__.py").is_file():
        raise RuntimeError("PhysicalRSI source is missing")


def reference_sources():
    # Bind the actual proposer, bridge and Self-Harness implementation. Exclude
    # this adapter (bound separately), release snapshots and unused providers.
    files = {}
    for relative in ("PhysicalRSI_core", "PhysicalRSI_baselines/robodojo",
                     "PhysicalRSI/Embodied_Harness"):
        base = RSI / relative
        for path in sorted(base.rglob("*.py")):
            if any(part in {"releases", "XPolicyLab", "__pycache__"} for part in path.parts):
                continue
            if path.is_symlink():
                raise ValueError("Reference source cannot be a symlink")
            files["reference-source/" + str(path.relative_to(RSI))] = path
    return files
