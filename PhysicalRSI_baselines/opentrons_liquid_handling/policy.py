"""Reviewed System 1 code policies; no robot connection or arbitrary code loading."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    wells: int = 6
    volume: int = 100

    def __post_init__(self):
        if type(self.wells) is not int or not 2 <= self.wells <= 12:
            raise ValueError('Expected 2–12 dilution wells')
        if type(self.volume) is not int or not 30 <= self.volume <= 150:
            raise ValueError('Expected 30–150 uL final volume')


CANDIDATES = {
    'unmixed': {'mix_repetitions': 0, 'batch_water': False},
    'mixed': {'mix_repetitions': 3, 'batch_water': False},
    'batched': {'mix_repetitions': 3, 'batch_water': True},
}


def policy_settings(settings):
    """Validate the bounded research domain without accepting executable input."""
    allowed = {'mix_repetitions', 'batch_water', 'reuse_mix_tip'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('Unknown policy setting')
    result = dict(mix_repetitions=3, batch_water=False, reuse_mix_tip=False)
    result.update(settings)
    if type(result['mix_repetitions']) is not int or not 0 <= result['mix_repetitions'] <= 10:
        raise ValueError('Expected 0–10 mixing repetitions')
    if any(type(result[k]) is not bool for k in ('batch_water', 'reuse_mix_tip')):
        raise ValueError('Expected boolean policy settings')
    return result


def plan(case, candidate):
    """Combine transfer, mix and water-distribution skills into a finite plan."""
    settings = policy_settings(CANDIDATES[candidate] if isinstance(candidate, str) else candidate)
    actions = []
    def emit(op, **kwargs):
        actions.append(dict(op=op, **kwargs))
    def transfer(source, target, volume, keep_tip=False):
        emit('pick_up_tip')
        emit('aspirate', well=source, volume=volume)
        emit('dispense', well=target, volume=volume)
        if not keep_tip:
            emit('drop_tip')
    wells = [f'A{i+1}' for i in range(case.wells)]
    if settings['batch_water']:
        batch = 300 // case.volume
        for offset in range(0, len(wells), batch):
            group = wells[offset:offset+batch]
            emit('pick_up_tip')
            emit('aspirate', well='water', volume=case.volume*len(group))
            for well in group:
                emit('dispense', well=well, volume=case.volume, above=True)
            emit('drop_tip')
    else:
        for well in wells:
            transfer('water', well, case.volume)
    for index, well in enumerate(wells):
        reuse = settings['reuse_mix_tip'] and bool(settings['mix_repetitions'])
        transfer('stock' if index == 0 else wells[index-1], well, case.volume, keep_tip=reuse)
        if settings['mix_repetitions']:
            if not reuse:
                emit('pick_up_tip')
            emit('mix', well=well, volume=case.volume, repetitions=settings['mix_repetitions'])
            emit('drop_tip')
    transfer(wells[-1], 'waste', case.volume)
    return actions


def protocol_source(actions):
    """Emit a standalone OT-2 protocol using explicit elementary API commands."""
    lines = ["metadata = {'apiLevel': '2.19', 'protocolName': 'physicalRSI dilution'}",
             'def run(protocol):',
             "    tips = protocol.load_labware('opentrons_96_tiprack_300ul', '1')",
             "    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '2')",
             "    reservoir = protocol.load_labware('nest_12_reservoir_15ml', '3')",
             "    pipette = protocol.load_instrument('p300_single_gen2', 'right', tip_racks=[tips])",
             "    wells = dict(plate.wells_by_name(), water=reservoir['A1'], stock=reservoir['A2'], waste=reservoir['A3'])"]
    for action in actions:
        op = action['op']
        if op in ('pick_up_tip', 'drop_tip'):
            lines.append(f'    pipette.{op}()')
        elif op in ('aspirate', 'dispense'):
            location = f"wells[{action['well']!r}]"
            if action.get('above'):
                location += '.top(2)'
            lines.append(f"    pipette.{op}({action['volume']!r}, {location})")
        elif op == 'mix':
            lines.append(f"    pipette.mix({action['repetitions']!r}, {action['volume']!r}, wells[{action['well']!r}])")
        else:
            raise ValueError('Unsupported primitive')
    return '\n'.join(lines)+'\n'
