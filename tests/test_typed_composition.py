import pytest

from PhysicalRSI_core.contracts import Contract, contract_from, contract_payload, mismatches
from physicalrsi import (CallableSkill, CompositionTypeError, SkillDescriptor,
                         SkillRegistry, adapt, check_composition, compose_sequence)


def skill(name, input_contract, output_contract, function=lambda value, context: value,
          metadata=None):
    return CallableSkill(SkillDescriptor(name, "r1", input_contract, output_contract,
                                         metadata=metadata), function)


def test_typed_sequence_and_node_kinds():
    number = Contract("number")
    primitive = skill("a", number, number)
    adapter = adapt("adapt", number, number, lambda value, context: value)
    inner = compose_sequence("inner", primitive, primitive)
    outer = compose_sequence("outer", primitive, adapter, inner)
    assert [node["kind"] for node in outer.composition.as_dict()["nodes"]] == [
        "primitive", "adapter", "composite"
    ]


def test_mismatch_error_names_steps_and_field():
    first = skill("producer", Contract("a"), Contract("b", unit="m"))
    second = skill("consumer", Contract("b", unit="cm"), Contract("c"))
    with pytest.raises(CompositionTypeError) as error:
        compose_sequence("bad", first, second)
    message = str(error.value)
    assert "producer" in message and "consumer" in message and "unit" in message


def test_contract_mismatch_rules():
    assert mismatches(Contract("x", unit="m"), Contract("x")) == []
    assert mismatches(Contract("x"), Contract("x", unit="m"))
    assert mismatches(Contract("x", shape=(480, 640, 3)),
                      Contract("x", shape=(None, None, 3))) == []
    assert mismatches(Contract("x", shape=(480, 640, 3)), Contract("x", shape=(3,)))
    assert mismatches(Contract("x"), Contract("x", shape=(3,)))
    assert mismatches(Contract("x", shape=(None,)), Contract("x", shape=(3,)))
    assert mismatches(Contract("x", dtype="uint8"), Contract("x", dtype="float32"))
    assert mismatches(Contract("x", rate_hz=30), Contract("x", rate_hz=60))


def test_adapter_executes_and_strict_false():
    source = Contract("source", dtype="uint8")
    target = Contract("target", dtype="float32")
    first = skill("first", source, source, lambda value, context: value + 1)
    second = skill("second", target, target, lambda value, context: value * 2)
    with pytest.raises(CompositionTypeError):
        compose_sequence("strict", first, second)
    assert compose_sequence("loose", first, second, strict=False).execute(1) == 4
    bridge = adapt("bridge", source, target, lambda value, context: float(value))
    assert compose_sequence("adapted", first, bridge, second).execute(1) == 4


def test_descriptor_registry_and_contract_round_trip(tmp_path):
    first = skill("typed", {"name": "x", "shape": (2, 3), "dtype": "float32", "rate_hz": 10},
                  Contract("x", shape=(2, 3), dtype="float32", rate_hz=10))
    second = skill("typed", {"name": "x", "shape": [2, 3], "dtype": "float32", "rate_hz": 10},
                   Contract("x", shape=[2, 3], dtype="float32", rate_hz=10))
    assert first.descriptor.as_dict() == second.descriptor.as_dict()
    registry = SkillRegistry(tmp_path)
    assert registry.register(first) == registry.register(second)
    key = registry.register(first)
    assert registry.load(key, implementation=first).descriptor == first.descriptor
    contract = Contract("x", unit="m", frame="world", embodiment="arm",
                        shape=(2, None), dtype="float32", rate_hz=10)
    assert contract_from(contract_payload(contract)) == contract


def test_check_composition_empty_when_compatible():
    contract = Contract("x")
    assert check_composition(skill("a", contract, contract), skill("b", contract, contract)) == {}
