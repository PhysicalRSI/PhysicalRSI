from functools import partial
import json
import os
import subprocess
import sys

import pytest

from PhysicalRSI_core.contracts import Context
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.storage import atomic_json, read_json
from PhysicalRSI_core.skill_graph.schema import SkillComposition
from physicalrsi import CallableSkill, SkillDescriptor, SkillRegistry, compose_sequence


def skill(name, function, *, revision='r1', effects=()):
    return CallableSkill(SkillDescriptor(name, revision, {'name': 'number'}, {'name': 'number'}, effects), function)


def positional(amount):
    return lambda value, context, amount=amount: value + amount


def keyword(amount):
    return lambda value, context, *, amount=amount: value + amount


def closure(amount):
    return lambda value, context: value + amount


def add(value, context, amount):
    return value + amount


class Offset:
    def __init__(self, amount):
        self.amount = amount

    def execute(self, value, context):
        return value + self.amount


class SlottedOffset:
    __slots__ = ('amount',)

    def __init__(self, amount):
        self.amount = amount

    def __call__(self, value, context):
        return value + self.amount


@pytest.mark.parametrize('factory', [positional, keyword, closure,
    lambda amount: partial(add, amount=amount),
    lambda amount: Offset(amount).execute, SlottedOffset])
def test_implementation_configuration_has_distinct_persistent_versions(tmp_path, factory):
    first, second = (skill('offset', factory(amount)) for amount in (1, 2))
    assert [first.execute(0), second.execute(0)] == [1, 2]
    assert first.executable_identity() != second.executable_identity()
    registry = SkillRegistry(tmp_path)
    first_key, second_key = registry.register(first), registry.register(second)
    assert first_key != second_key
    restarted = SkillRegistry(tmp_path)
    assert restarted.register(skill('offset', factory(1))) == first_key
    assert restarted.register(skill('offset', factory(2))) == second_key
    assert restarted.load(first_key, implementation=first).execute(0) == 1
    assert restarted.load(second_key, implementation=second).execute(0) == 2


def test_nested_function_identity_is_stable_across_processes(tmp_path):
    script = '''
import json
from physicalrsi import CallableSkill, SkillDescriptor
class Parameters:
    def __init__(self):
        self.amount = 2
        self.owner = self
def build():
    parameters = Parameters()
    def execute(value, context):
        return sum(item + parameters.amount for item in range(value))
    return CallableSkill(SkillDescriptor('sum', 'r1', {}, {}), execute)
value = build()
print(json.dumps({'identity': value.executable_identity(), 'result': value.execute(3)}, sort_keys=True))
'''
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), PYTHONDONTWRITEBYTECODE='1')
    def run():
        return subprocess.check_output([sys.executable, '-B', '-c', script], env=environment, text=True)
    assert run() == run()
    assert json.loads(run())['result'] == 9


def test_explicit_implementation_identity_remains_provider_owned():
    declared = {'provider': 'robot-controller', 'version': 'v2', 'parameters': {'gain': 0.5}}
    controller = CallableSkill(SkillDescriptor('move', 'v2', {}, {}),
                               lambda value, context: value, executable_identity=declared)
    assert controller.executable_identity() == declared
    assert controller.execute({'arbitrary': 'input'}) == {'arbitrary': 'input'}


def test_nested_composition_preserves_order_and_implementation_changes(tmp_path):
    inc = skill('inc', positional(1))
    double = skill('double', lambda value, context: value * 2)
    a = compose_sequence('inner', inc, double)
    b = compose_sequence('inner', double, inc)
    outer_a, outer_b = (compose_sequence('outer', inner) for inner in (a, b))
    assert [outer_a.execute(1), outer_b.execute(1)] == [4, 3]
    assert a.revision != b.revision
    assert outer_a.composition_identity() != outer_b.composition_identity()
    assert outer_a.composition.nodes[0].kind == 'composite'
    assert outer_a.composition.nodes[0].composition['nodes'][0]['skill'] == 'inc'
    changed = compose_sequence('outer', compose_sequence('inner', skill('inc', positional(2)), double))
    assert changed.execute(1) == 6
    assert changed.composition_identity() != outer_a.composition_identity()
    assert SkillComposition.from_dict(outer_a.composition_identity()).as_dict() == outer_a.composition_identity()
    registry = SkillRegistry(tmp_path)
    keys = [registry.register(item) for item in (outer_a, outer_b, changed)]
    assert len(set(keys)) == 3
    restored = SkillRegistry(tmp_path).load(keys[0], implementation=outer_a)
    assert restored.execute(1) == 4
    assert restored.composition_identity() == outer_a.composition_identity()


def test_nested_shared_effects_and_operation_runtime_keep_all_steps(tmp_path):
    a = skill('a', positional(1), effects=('robot.arm',))
    b = skill('b', positional(2), effects=('robot.arm',))
    c = skill('c', positional(3), effects=('robot.arm', 'robot.gripper'))
    inner = compose_sequence('inner', a, b)
    outer = compose_sequence('outer', inner, c)
    assert outer.descriptor.effects == ('robot.arm', 'robot.gripper')
    assert outer.execute(0) == 6
    operation = outer.as_operation()
    assert [child.name for child in operation.children] == ['inner', 'c']
    assert [child.name for child in operation.children[0].children] == ['a', 'b']
    calls = []
    def execute(op, value, context):
        calls.append(op.name)
        return op.call(value, context)
    assert operation(0, Context('nested', execution=execute)) == 6
    assert calls == ['outer', 'inner', 'a', 'b', 'c']
    assert operation(0, Context('recorded', execution=Execution(tmp_path))) == 6
    records = [read_json(path) for path in (tmp_path / 'recorded').glob('*.json')]
    assert {row['operation'] for row in records} == set(calls)
    assert all(row['state'] == 'completed' for row in records)


def test_registry_binding_reports_actual_provider_and_preserves_saved_record(tmp_path):
    original = compose_sequence('original', skill('inc', positional(1)))
    replacement = compose_sequence('replacement', skill('double', lambda value, context: value * 2))
    registry = SkillRegistry(tmp_path)
    key = registry.register(original)
    path = tmp_path / 'skills' / (key + '.json')
    before = path.read_bytes()
    unbound = SkillRegistry(tmp_path).load(key)
    assert unbound.stored_record() == original.registry_record()
    assert unbound.executable_identity() is None
    assert unbound.execution_identity() is None
    assert unbound.binding_identity()['implementation'] is None
    assert registry.register(unbound) == key
    with pytest.raises(RuntimeError, match='trusted implementation'):
        unbound.execute(1)
    rebound = registry.load(key, implementation=replacement)
    assert rebound.execute(3) == 6
    assert rebound.identity() == replacement.identity()
    assert rebound.executable_identity() == replacement.executable_identity()
    assert rebound.execution_identity() == replacement.executable_identity()
    assert rebound.composition_identity() == replacement.composition_identity()
    assert rebound.binding_identity()['record'] == original.registry_record()
    assert rebound.binding_identity()['implementation'] == replacement.registry_record()
    rebound.stored_record()['descriptor']['id'] = 'changed-copy'
    assert rebound.stored_record()['descriptor']['id'] == original.id
    rebound_key = registry.register(rebound)
    assert rebound_key != key
    assert SkillRegistry(tmp_path).load(rebound_key).stored_record() == replacement.registry_record()
    assert path.read_bytes() == before
    new_graph = compose_sequence('after-binding', rebound)
    assert new_graph.composition.nodes[0].skill == replacement.id
    assert new_graph.execute(3) == 6
    calls = []
    def execute(op, value, context):
        calls.append(op.name)
        return op.call(value, context)
    assert rebound.as_operation()(3, Context('rebound', execution=execute)) == 6
    assert calls == ['replacement', 'double']


def test_long_composition_uses_short_storage_key_and_old_logical_keys_still_load(tmp_path):
    primitive = skill('pose', positional(1), revision='x' * 64)
    composed = compose_sequence('four', primitive, primitive, primitive, primitive)
    registry = SkillRegistry(tmp_path)
    key = registry.register(composed)
    assert len(key.encode()) < 100
    assert registry.load(key).execute(0) == 4
    record = primitive.registry_record()
    old_key = 'pose@' + primitive.revision
    path = tmp_path / 'skills' / (old_key + '.json')
    atomic_json(path, record)
    before = path.read_bytes()
    restarted = SkillRegistry(tmp_path)
    assert restarted.register(primitive) == old_key
    assert restarted.load(old_key, implementation=primitive).execute(0) == 1
    assert path.read_bytes() == before


def test_registry_does_not_merge_boolean_and_numeric_parameters(tmp_path):
    def build(amount):
        return skill('typed-parameter', lambda value, context, amount=amount: type(amount).__name__)
    registry = SkillRegistry(tmp_path)
    numeric, boolean = registry.register(build(1)), registry.register(build(True))
    assert numeric != boolean
    assert registry.load(numeric).execute(None) == 'int'
    assert registry.load(boolean).execute(None) == 'bool'
