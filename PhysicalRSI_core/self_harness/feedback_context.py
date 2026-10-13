"""Comparable execution conditions for development-feedback search."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from ..infra.storage import digest


def measurement_context(value):
    """Pin measured dynamics, verifier, budget and contracts across fidelities.

    Policy parameters vary within the declared harness space. Their closure is
    checked separately; suite identity/admission attests policy construction.
    """
    paths=[p for p in value['evidence'] if Path(p).name=='experiment.json'
           and Path(p).parent.name==value['trial']['id']]
    if len(paths)!=1:
        raise ValueError('Development evaluation requires one bound experiment specification')
    raw=Path(paths[0]).read_bytes()
    if sha256(raw).hexdigest()!=value['evidence'][paths[0]]:
        raise ValueError('Development experiment specification changed')
    spec=json.loads(raw)
    if digest(spec)!=value['receipt']['experiment_sha256']:
        raise ValueError('Experiment specification differs from its Trial receipt')
    if any(spec[k]!=value['trial'][k] for k in ('task','case','scope','budget','system2')):
        raise ValueError('Experiment specification differs from the attributed Trial')
    adapters={k:v for k,v in spec['adapters'].items() if k!='policy'}
    binding=deepcopy(spec.get('binding'))
    if binding is not None:
        binding['system1'].pop('revision',None)
    return digest(dict(adapters=adapters,binding=binding,budget=spec['budget'],
        artifact_contract=spec.get('artifact_contract'),policy_lifecycle=spec.get('policy_lifecycle')))
