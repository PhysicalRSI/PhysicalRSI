import pytest

from PhysicalRSI_baselines.embodied_goodharts_law.skill_memory_dependencies import inspect_skill_memory


def test_transitive_required_reads_ignore_unused_helpers_and_optional_get():
    source = '''
def grasp(robot, memory):
    locate(robot, memory)
    return memory.get('optional', 0)
def locate(robot, memory):
    return memory['support_prompts']
def unused(robot, memory):
    return memory['unrelated']
'''
    d = inspect_skill_memory(source, {}, entry_points=['grasp'])
    assert d['missing_keys'] == ['support_prompts']
    assert d['visited_functions'] == ['grasp', 'locate']
    assert not d['complete_dependency_analysis']
    assert not d['execution_validated']


def test_recursive_helpers_terminate_and_existing_keys_pass():
    d = inspect_skill_memory("def f(robot, memory):\n f(robot, memory)\n return memory['x']", {'x': 1}, entry_points=['f'])
    assert d['missing_keys'] == []
    assert not d['execution_validated']


def test_missing_entry_point_is_not_silently_accepted():
    with pytest.raises(ValueError, match='entry point'):
        inspect_skill_memory('def other(): pass', {}, entry_points=['grasp'])
