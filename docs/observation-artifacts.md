# Verified local observation artifacts

`infra.artifacts.Artifacts` stores binary observations and NumPy arrays separately from compact JSON execution records. `Execution` already uses it for operation inputs, outputs and post-action observations. The reference retains its absolute local path, SHA256 and encoded byte count.

Consumers can now verify references before using their contents:

```python
import tempfile
import numpy as np
from PhysicalRSI_core.infra.artifacts import Artifacts

store = Artifacts(tempfile.mkdtemp(prefix="physicalrsi-frames-"))
reference = store.encode(np.zeros((480, 640, 3), dtype=np.uint8))
frame = store.read_array(reference, max_bytes=2 * 1024 * 1024)
assert frame.shape == (480, 640, 3)
assert not frame.flags.writeable

binary = store.encode(b"sensor-packet")
assert store.read(binary, max_bytes=1024) == b"sensor-packet"
```

`read()` accepts only the reference fields shown above and a content-addressed `.bin` or `.npy` filename directly inside the configured store. The caller supplies a nonnegative byte allowance. The reader checks that allowance before opening, refuses a final symlink, checks the opened descriptor is a regular file of the declared size, reads at most that size plus one byte, and verifies the digest. Missing, changed, oversized or foreign-store evidence fails; it is never silently reconstructed from another path. FIFO and device inputs do not become blocking stream reads.

`read_array()` first performs those checks. It supports NPY versions 1 and 2, preserves dtype and C/Fortran layout, and returns a read-only view owned by the verified bytes. It limits headers to 10,000 bytes, rejects object and zero-itemsize dtypes, and checks nonnegative shape dimensions and the exact data byte count before constructing the view. Truncated data and trailing bytes are rejected. It never calls a pickle loader. NumPy documents the [bounded header reader](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.read_array_header_2_0.html) and the [risks of loading pickled object arrays](https://numpy.org/doc/stable/reference/generated/numpy.load.html). Other formats remain raw bytes for an explicitly chosen decoder; no decoder is inferred from model output.

Publication writes and syncs a temporary file, then hard-links it into its content-addressed destination without replacing an existing entry. An existing entry is verified, including after a concurrent publication, and corrupt evidence is left intact with an error. The store directory is synced before returning the reference. Create and protect the store and its ancestor directories as part of trusted workspace provisioning. This requires local POSIX filesystem semantics; it is not an isolation boundary against a hostile host writer.

The allowance bounds bytes read for one artifact, not total process RAM, all frames in a batch or filesystem latency. Array decoding retains the verified byte buffer and header parsing has its own limit. Producers must separately bound capture and encoding. The API does not fetch remote URLs, authorize another host, translate paths between machines, expose store files to an isolated candidate, validate sensor calibration, or establish observation freshness. Those responsibilities belong to the configured transport, execution profile and observation contract. Reading valid bytes does not establish physical task success.

## Inspect a complete operation record

`Execution.read(episode, call_id)` checks the execution journal and every recognized artifact reference nested in its `input`, `output` and `observation` fields. It returns the original record, the digest of the exact journal bytes, a sorted artifact manifest and the total encoded bytes verified. The API invokes no operation, observer, model or event callback.

```python
inspection = execution.read(
    episode_id,
    call_id,
    max_record_bytes=1024 * 1024,
    max_artifact_bytes=64 * 1024 * 1024,
    expected_sha256=saved_record_digest,
)
record = inspection["record"]
```

The JSON and cumulative artifact allowances are separate. A repeated identical reference is verified once; conflicting declarations for one path are rejected. Journal identity, state, duplicate JSON members, nonfinite numbers, path traversal and symlinked episode directories are checked before returning an inspection. References remain references; array decoding is a separate explicit consumer operation through `read_array()`.

Supply `expected_sha256` from an independently retained receipt when verifying an earlier snapshot. Without that argument the API inspects the current trusted journal and reports its digest; it does not authenticate past metadata. A live `started` record remains started, and failed, cancelled or uncertain records keep their original labels. Successful inspection means the inspected bytes satisfy these integrity checks, not that an operation succeeded, a device is quiescent, or a retry is authorized. The store and journal directories require the same trusted-host protection described above. This API serves the operation `Execution` layer; `ExperimentRuntime` trial receipts have their own evidence and recovery contract.

Run the local evidence checks with:

```bash
python -m pytest -q tests/test_artifact_reads.py tests/test_execution_inspection.py
```

These exercise an actual image-producing operation record, immutable snapshots, scalar/empty/Fortran arrays, concurrent publication, changed content, foreign paths, symlinks, FIFOs, malformed allocation headers and both supported NPY versions. They do not use a camera, model provider or robot.
