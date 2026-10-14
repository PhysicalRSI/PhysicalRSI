"""Sequential bounded host reads for pinned Orbax Zarr/OCDBT array checkpoints.

This experimental path requires one destination device. It does not alter shape,
checkpoint contents. An explicit destination dtype uses NumPy conversion,
matching Orbax host restoration; parity must be verified before native use. TensorStore may decode a storage chunk larger than
the requested tile, so the tile bound is not a claim about total process RSS.
"""
from pathlib import Path
import itertools
import functools
import math
import numpy as np
import jax
import jax.numpy as jnp
import tensorstore as ts
from orbax.checkpoint._src.serialization.serialization import get_tensorstore_spec
from orbax.checkpoint._src.tree import utils as tree_utils


def restore_arrays(path, metadata, *, sharding, chunk_bytes, dtype=None):
    if type(chunk_bytes) is not int or chunk_bytes<=0:
        raise ValueError('Positive integer tile byte bound required')
    if sharding is None or len(sharding.device_set)!=1:
        raise ValueError('Chunk restore currently requires one explicit destination device')
    path=Path(path).resolve()
    if not (path/'manifest.ocdbt').is_file():
        raise ValueError('This path requires an Orbax OCDBT checkpoint')
    names=tree_utils.get_param_names(metadata)
    context=ts.Context({'cache_pool':{'total_bytes_limit':0},'data_copy_concurrency':{'limit':2}})
    @functools.partial(jax.jit,donate_argnums=(0,))
    def insert(target, block, start):
        return jax.lax.dynamic_update_slice(target,block,start)
    def read(name, info):
        spec=get_tensorstore_spec(str(path/name),ocdbt=True)
        spec['fill_missing_data_reads']=False
        store=ts.open(spec,read=True,write=False,context=context).result()
        shape=tuple(info.shape);source_dtype=np.dtype(info.dtype)
        target_dtype=np.dtype(dtype) if dtype is not None else source_dtype
        if tuple(store.shape)!=shape or store.dtype.numpy_dtype!=source_dtype:
            raise ValueError('Stored shape or dtype differs from checkpoint metadata')
        if source_dtype.itemsize>chunk_bytes or any(x<=0 for x in shape):
            raise ValueError('Unsupported array dimensions or tile bound')
        tile=list(shape)
        while math.prod(tile)*source_dtype.itemsize>chunk_bytes:
            axis=max(range(len(tile)),key=tile.__getitem__)
            tile[axis]=(tile[axis]+1)//2
        if not shape:
            host=store.read().result().astype(target_dtype);value=jax.device_put(host,sharding)
            if value.dtype!=target_dtype or np.asarray(jax.device_get(value)).tobytes()!=host.tobytes():
                raise ValueError('Scalar restoration changed checkpoint bytes')
            return value
        target=jax.jit(lambda:jnp.zeros(shape,target_dtype),out_shardings=sharding)()
        if target.dtype!=target_dtype:
            raise ValueError('Destination dtype differs from checkpoint')
        for start in itertools.product(*(range(0,n,c) for n,c in zip(shape,tile,strict=True))):
            slices=tuple(slice(s,min(s+c,n)) for s,c,n in zip(start,tile,shape,strict=True))
            host=store[slices].read().result()
            if host.nbytes>chunk_bytes:
                raise ValueError('Read exceeds declared host tile bound')
            host=host.astype(target_dtype)
            block=jax.device_put(host,sharding)
            target=insert(target,block,tuple(jnp.int32(s) for s in start))
            target.block_until_ready()
            received=np.asarray(jax.device_get(target[slices]))
            if received.tobytes()!=host.tobytes():
                raise ValueError('Device tile differs from checkpoint bytes')
            del host,block,received
        return target
    return jax.tree.map(read,names,metadata)
