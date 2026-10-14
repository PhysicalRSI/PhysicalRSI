"""Experimental local single-device equivalent of openpi.restore_params."""
from pathlib import Path
import jax
import orbax.checkpoint as ocp
from flax import traverse_util
from .champion_chunk_restore import restore_arrays

def restore_params(params_path, *, dtype=None, sharding=None, chunk_bytes=64*1024**2):
    path=Path(params_path).resolve()
    if sharding is None:
        mesh=jax.sharding.Mesh(jax.devices(),('x',))
        sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckpt:
        metadata=ckpt.metadata(path)
        params=restore_arrays(path,{'params':metadata['params']},sharding=sharding,chunk_bytes=chunk_bytes,dtype=dtype)['params']
    flat=traverse_util.flatten_dict(params)
    if all(kp[-1]=='value' for kp in flat):
        flat={kp[:-1]:v for kp,v in flat.items()}
    return traverse_util.unflatten_dict(flat)
