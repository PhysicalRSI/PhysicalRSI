from pathlib import Path

from PhysicalRSI_baselines.robodojo.xpolicy_package import install


def test_adapter_installs_into_an_upstream_checkout(tmp_path):
    checkout = tmp_path / "XPolicyLab"
    checkout.mkdir()
    (checkout / "model_template.py").write_text("class ModelTemplate: pass\n")

    destination = install(checkout)

    required = {
        "README.md",
        "__init__.py",
        "install.sh",
        "eval.sh",
        "setup_eval_policy_server.sh",
        "setup_eval_env_client.sh",
        "deploy.yml",
        "deploy.py",
        "model.py",
        "code_policy_demo.py",
    }
    assert destination == checkout / "policy" / "physicalRSI"
    assert required <= {path.name for path in destination.iterdir()}
    assert (destination / 'download_assets.py').is_file()
    assert not (destination / 'assets.json').exists()
    import re
    obsolete = re.compile('route|router|routing', re.IGNORECASE)
    for path in destination.rglob('*'):
        assert not obsolete.search(str(path.relative_to(destination))), path
        if path.is_file() and path.suffix in {'.py', '.json', '.md', '.sh', '.yml', '.yaml'}:
            assert not obsolete.search(path.read_text()), path
    assert (destination / "install.sh").stat().st_mode & 0o111
    assert not (destination / "train.sh").exists()
    assert not (destination / "process_data.sh").exists()


def test_installed_runtime_imports_without_source_checkout(tmp_path):
    import os
    import subprocess
    import sys
    checkout = tmp_path / 'XPolicyLab'
    checkout.mkdir()
    (checkout / 'model_template.py').write_text('class ModelTemplate: pass\n')
    destination = install(checkout)
    environment = dict(os.environ, PYTHONPATH=str(destination / 'runtime'))
    result = subprocess.run(
        [sys.executable, '-c', 'from PhysicalRSI_baselines.robodojo.skill_model import Model; from PhysicalRSI_baselines.robodojo.execution_skills import compose_skill; print("runtime-import-ok")'],
        cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'runtime-import-ok' in result.stdout
    assert (destination / 'runtime/PhysicalRSI_baselines/robodojo/configs/skill_compositions.json').is_file()
    assert not (destination / 'runtime/PhysicalRSI_baselines/robodojo/trained_policy').exists()
    assert not (destination / 'runtime/PhysicalRSI_baselines/dexterous').exists()
    assert (destination / 'runtime/PhysicalRSI_baselines/robodojo/skills/LICENSE.openpi').is_file()


def test_installed_entrypoint_inherits_template_and_uses_skill_lifecycle(tmp_path):
    import os
    import subprocess
    import sys
    checkout = tmp_path / 'XPolicyLab'
    checkout.mkdir()
    (checkout / 'model_template.py').write_text(
        'class ModelTemplate:\n'
        '    def __init__(self): self.template_initialized = True\n'
        '    def get_action(self): raise NotImplementedError\n')
    destination = install(checkout)
    code = """
from XPolicyLab.model_template import ModelTemplate
from PhysicalRSI_baselines.robodojo.skill_model import Model as SkillModel
from XPolicyLab.policy.physicalRSI.model import Model
assert issubclass(Model, ModelTemplate)
assert Model.__bases__ == (ModelTemplate,)
original = SkillModel.__init__
try:
    SkillModel.__init__ = lambda self, config: setattr(self, 'received_config', config)
    instance = Model({'check': True})
    assert instance.template_initialized
    assert instance.model.received_config == {'check': True}
    SkillModel.get_action = lambda self: self.received_config
    assert instance.get_action() == {'check': True}
finally:
    SkillModel.__init__ = original
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path),str(destination/'runtime')])),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
