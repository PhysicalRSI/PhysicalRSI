"""Run only the upstream software simulator in a separate pinned interpreter."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('protocol',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    version=importlib.metadata.version('opentrons')
    if version != '8.8.2': raise ValueError('Expected opentrons==8.8.2')
    config = a.output.parent / 'opentrons-config'
    config.mkdir(exist_ok=False)
    os.environ['OT_API_CONFIG_DIR'] = str(config.resolve())
    from opentrons import simulate
    with a.protocol.open() as handle:
        log,_=simulate.simulate(handle)
    # Upstream payloads contain Well/Location objects; preserve their textual
    # representation rather than pretending they are portable robot states.
    a.output.write_text(json.dumps({'scope':'opentrons-software-simulation','qualification':None,
        'opentrons_version':version,'commands':log},default=str,indent=2)+'\n')


if __name__ == '__main__': main()
