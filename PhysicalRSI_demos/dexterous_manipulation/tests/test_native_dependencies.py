from pathlib import Path

import pytest

from PhysicalRSI_demos.dexterous_manipulation.native_dependencies import NativeDependencyClosure


@pytest.mark.parametrize("change",["asset","new_module","configuration"])
def test_native_closure_detects_non_entrypoint_changes(tmp_path,change):
    package=tmp_path/"dexjoco/dexjoco"
    package.mkdir(parents=True)
    asset=package/"mesh.obj"
    asset.write_text("fixture mesh")
    (package/"__init__.py").write_text("")
    config=tmp_path/"configs/rand_obj/fixture.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("task: fixture\n")
    closure=NativeDependencyClosure(tmp_path)
    assert closure.check(hash_bytes=True)==closure.revision
    if change=="asset": asset.write_text("changed mesh")
    elif change=="configuration": config.write_text("task: changed\n")
    else: (package/"new.py").write_text("changed = True\n")
    with pytest.raises(ValueError,match="Native source or asset inventory changed"):
        closure.check()
