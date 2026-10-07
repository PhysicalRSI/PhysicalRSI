from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from PhysicalRSI_core.contracts import Context, Contract, Operation
from PhysicalRSI_core.infra.artifacts import Artifacts
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.storage import read_json


def reference(store, content, suffix=".npy"):
    sha = hashlib.sha256(content).hexdigest()
    store.root.mkdir(parents=True, exist_ok=True)
    path = store.root / (sha + suffix)
    path.write_bytes(content)
    return dict(artifact=str(path), sha256=sha, bytes=len(content))


def test_operation_image_evidence_can_be_verified_and_read_after_recording(tmp_path):
    execution = Execution(tmp_path)
    frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    operation = Operation("camera-filter", "1", Contract("rgb"), Contract("rgb"), lambda value, ctx: value[:, ::-1])
    result = execution(operation, frame, Context("episode"))
    record = read_json(next((tmp_path / "episode").glob("*.json")))
    observed = execution.artifacts.read_array(record["input"], max_bytes=1024)
    filtered = execution.artifacts.read_array(record["output"], max_bytes=1024)
    assert np.array_equal(observed, frame) and np.array_equal(filtered, result)
    frame[:] = 0
    assert observed.sum() > 0 and not observed.flags.writeable
    with pytest.raises(ValueError):
        observed[0, 0, 0] = 1


@pytest.mark.parametrize("array", [np.array(3.5), np.empty((0, 3)),
    np.asfortranarray(np.arange(12, dtype=">i4").reshape(3, 4))])
def test_array_layout_dtype_and_empty_dimensions_are_preserved(tmp_path, array):
    store = Artifacts(tmp_path)
    value = store.read_array(store.encode(array), max_bytes=4096)
    assert value.shape == array.shape and value.dtype == array.dtype
    assert np.array_equal(value, array) and not value.flags.writeable


def test_concurrent_publication_reuses_content_without_overwriting_corruption(tmp_path):
    store = Artifacts(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(store.encode, [b"camera-frame"] * 12))
    assert all(ref == refs[0] for ref in refs)
    assert store.read(refs[0], max_bytes=12) == b"camera-frame"
    path = Path(refs[0]["artifact"])
    path.write_bytes(b"broken-frame")
    with pytest.raises(ValueError, match="differs"):
        store.encode(b"camera-frame")
    assert path.read_bytes() == b"broken-frame"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", ["budget", "outside", "length", "content", "symlink", "fifo"])
def test_read_rejects_untrusted_locations_content_and_unbounded_input(tmp_path, change):
    store = Artifacts(tmp_path / "store")
    ref = store.encode(b"abc")
    path = Path(ref["artifact"])
    limit = 3
    if change == "budget":
        limit = 2
    elif change == "outside":
        other = tmp_path / path.name
        other.write_bytes(b"abc")
        ref["artifact"] = str(other)
    elif change == "length":
        ref["bytes"] = 2
    elif change == "content":
        path.write_bytes(b"xyz")
    elif change == "symlink":
        other = tmp_path / "outside"
        other.write_bytes(b"abc")
        path.unlink()
        path.symlink_to(other)
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        store.read(ref, max_bytes=limit)


@pytest.mark.parametrize("kind", ["huge_shape", "negative_shape", "object", "trailing", "header", "version"])
def test_array_headers_cannot_request_unverified_allocations_or_pickle(tmp_path, monkeypatch, kind):
    store = Artifacts(tmp_path)
    stream = io.BytesIO()
    dtype = np.dtype("O") if kind == "object" else np.dtype("u1")
    shape = (10**12,) if kind == "huge_shape" else (1,)
    if kind == "negative_shape":
        shape = (-1, -1)
    np.lib.format.write_array_header_1_0(stream, dict(descr=dtype.str, shape=shape, fortran_order=False))
    content = stream.getvalue() + b"x"
    if kind == "trailing":
        content += b"trailing"
    elif kind == "header":
        header = b" " * 11000
        content = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    elif kind == "version":
        content = content[:6] + b"\x03\x00" + content[8:]
    monkeypatch.setattr(np, "frombuffer", lambda *a, **k: pytest.fail("Bad header reached array construction"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Artifact reader must not use pickle-capable loading"))
    with pytest.raises(ValueError):
        store.read_array(reference(store, content), max_bytes=20000)


def test_empty_bytes_have_a_valid_zero_byte_allowance(tmp_path):
    store = Artifacts(tmp_path)
    assert store.read(store.encode(b""), max_bytes=0) == b""


def test_npy_version_two_is_read_without_changing_dtype(tmp_path):
    store = Artifacts(tmp_path)
    stream = io.BytesIO()
    np.lib.format.write_array_header_2_0(stream, dict(descr="|u1", shape=(1,), fortran_order=False))
    value = store.read_array(reference(store, stream.getvalue() + b"\x07"), max_bytes=1024)
    assert value.tolist() == [7] and value.dtype == np.dtype("u1")
