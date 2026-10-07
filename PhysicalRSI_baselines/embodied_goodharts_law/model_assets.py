"""Inventory external files referenced by a trusted compiled MuJoCo model."""
from pathlib import Path
import xml.etree.ElementTree as ET

from PhysicalRSI_core.infra.storage import file_digest


def model_asset_inventory(xml, *, allowed_roots):
    """Hash model meshes/textures without treating XML paths as read authority.

    Compiled native models currently contain absolute asset references. Reject
    ambiguous relative references instead of guessing the compiler's directory.
    This covers referenced assets, not the simulator's full dependency closure.
    """
    roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("Declare prepared asset directories")
    files = {}
    for element in ET.fromstring(xml).iter():
        reference = element.get("file")
        if reference is None:
            continue
        if element.tag not in {"mesh", "texture", "hfield", "skin"}:
            raise ValueError("Unsupported compiled-model file reference")
        if not Path(reference).is_absolute():
            raise ValueError("Compiled asset reference is not absolute")
        path = Path(reference).resolve(strict=True)
        if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
            raise ValueError("Model asset lies outside the declared asset directories")
        if reference not in files:
            files[reference] = {"resolved_path": str(path), "bytes": path.stat().st_size,
                                "sha256": file_digest(path)}
    return {"schema": "physicalrsi.egl-model-assets/v1", "files": dict(sorted(files.items()))}
