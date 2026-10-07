"""Independent ideal mass-balance audit, not a fluid physics or assay model."""
import math


def audit(case, actions):
    # Concentration is relative to stock. Uniform mixing is an explicit assumption
    # only after a qualifying mix command. Aspirating an unmixed blend is rejected.
    wells = {f'A{i+1}': [0., 0., True] for i in range(case.wells)}
    wells.update(water=[10000., 0., True], stock=[1000., 1000., True], waste=[0., 0., True])
    tip = False
    held_volume = held_mass = 0.
    touched = None
    tips = 0
    errors = []
    try:
        for a in actions:
            op = a['op']
            if op == 'pick_up_tip':
                if tip: raise ValueError('Tip already attached')
                tip = True; tips += 1; touched = None
            elif op == 'drop_tip':
                if not tip or held_volume > 1e-8: raise ValueError('Missing or nonempty tip at drop')
                tip = False
            else:
                if not tip: raise ValueError('Liquid operation without tip')
                v = a['volume']
                if type(v) not in (int, float) or not math.isfinite(v) or not 20 <= v <= 300:
                    raise ValueError('Outside declared P300 working range')
                name = a['well']; well = wells[name]
                if op == 'aspirate':
                    if not well[2]: raise ValueError('Aspiration from unmixed dilution')
                    if well[0] < v or held_volume+v > 300: raise ValueError('Insufficient source or tip capacity')
                    if touched is not None and touched != name: raise ValueError('Cross-source tip contact')
                    mass = v*well[1]/well[0]
                    well[0] -= v; well[1] -= mass
                    held_volume += v; held_mass += mass; touched = name
                elif op == 'dispense':
                    if held_volume < v: raise ValueError('Dispense exceeds aspirated volume')
                    capacity = 15000 if name in ('water','stock','waste') else 360
                    if well[0]+v > capacity: raise ValueError('Well overflow')
                    mass = v*held_mass/held_volume
                    same = well[0] == 0 or math.isclose(well[1]/well[0], mass/v, abs_tol=1e-12)
                    well[2] = well[2] and same
                    well[0] += v; well[1] += mass
                    held_volume -= v; held_mass -= mass
                    if not a.get('above'): touched = name
                elif op == 'mix':
                    if held_volume > 1e-8 or well[0] < v: raise ValueError('Invalid mixing volume')
                    if type(a['repetitions']) is not int or a['repetitions'] < 3:
                        raise ValueError('Insufficient declared mixing repetitions')
                    well[2] = True; touched = name
                else:
                    raise ValueError('Unknown operation')
        if tip or held_volume > 1e-8: raise ValueError('Unfinished liquid operation')
        for i in range(case.wells):
            v,m,mixed = wells[f'A{i+1}']
            if not mixed or not math.isclose(v, case.volume, abs_tol=1e-8) or not math.isclose(m/v, 2**(-i-1), abs_tol=1e-10):
                raise ValueError('Final concentration, volume or homogeneity mismatch')
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        errors.append(str(exc))
    return {'passed': not errors, 'errors': errors, 'tips': tips,
            'primitive_commands': len(actions), 'wells': wells,
            'model': 'ideal-mass-balance-with-explicit-mixing', 'physical_measurement': False}
