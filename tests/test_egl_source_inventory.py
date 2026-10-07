import hashlib
import subprocess
from PhysicalRSI_baselines.embodied_goodharts_law.source_inventory import capture


def test_inventory_captures_modified_bytes_without_following_symlinks(tmp_path):
    def git(*args):
        return subprocess.check_output(['git','-C',str(tmp_path),*args])
    git('init','-q')
    (tmp_path/'module.py').write_bytes(b'original')
    (tmp_path/'removed.py').write_bytes(b'old')
    (tmp_path/'external').symlink_to('/unavailable/model-assets')
    git('add','.')
    git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    (tmp_path/'module.py').write_bytes(b'local modification')
    (tmp_path/'removed.py').unlink()
    (tmp_path/'untracked.py').write_bytes(b'extra')
    result=capture(tmp_path)
    rows={x['path']:x for x in result['tracked']}
    assert rows['module.py']['sha256']==hashlib.sha256(b'local modification').hexdigest()
    assert rows['external']['symlink_target']=='/unavailable/model-assets'
    assert not rows['external']['target_contents_inventoried']
    assert {'path':'removed.py','problem':'missing-tracked-file'} in result['issues']
    assert 'untracked.py' not in rows
    assert 'untracked.py' in result['status_porcelain']
    assert not result['complete_execution_closure_attested']


def test_uninitialized_submodule_does_not_inherit_parent_commit(tmp_path):
    def git(*args):
        return subprocess.check_output(['git','-C',str(tmp_path),*args])
    git('init','-q')
    (tmp_path/'module.py').write_bytes(b'fixture')
    git('add','.')
    git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')
    parent=git('rev-parse','HEAD').decode().strip()
    git('update-index','--add','--cacheinfo','160000',parent,'uninitialized')
    (tmp_path/'uninitialized').mkdir()
    result=capture(tmp_path)
    row=next(x for x in result['tracked'] if x['path']=='uninitialized')
    assert row['checkout_commit'] is None
    assert {'path':'uninitialized','problem':'submodule-not-initialized'} in result['issues']
