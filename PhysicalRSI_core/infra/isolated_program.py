"""Pinned, stdlib-only Linux execution for data-producing candidate programs.

The operator supplies a trusted runtime. Candidate source never runs in the
controller process. This profile provides no network or device capabilities.
"""

from copy import deepcopy
import ctypes.util
import os
from pathlib import Path
import sys

from . import isolation, isolation_bridge, isolation_entry, processes
from .storage import digest, file_digest, read_json
from ..timing import finite_seconds


class PythonIsolation:
    def __init__(self, runtime, *, memory_bytes=256 * 1024 * 1024, message_bytes=1024 * 1024, teardown_seconds=2):
        if sys.platform != "linux" or os.geteuid() != 0 or not ctypes.util.find_library("seccomp"):
            raise RuntimeError("Python isolation requires Linux chroot privilege and libseccomp; no fallback is available")
        if type(memory_bytes) is not int or memory_bytes <= 0 or type(message_bytes) is not int or message_bytes <= 0:
            raise ValueError("Positive integer memory and message limits required")
        finite_seconds(teardown_seconds)
        self.runtime = Path(runtime).resolve()
        self.memory_bytes, self.message_bytes, self.teardown_seconds = memory_bytes, message_bytes, teardown_seconds
        manifest = read_json(self.runtime / "manifest.json")
        if manifest.get("schema") != "physicalrsi.isolation-runtime/v1":
            raise ValueError("Unsupported isolation runtime")
        self.runtime_sha256 = digest(manifest)
        self._frozen = deepcopy(self.identity())

    def identity(self):
        return dict(kind="linux-python-data-isolation/v1", runtime_sha256=digest(read_json(self.runtime / "manifest.json")),
                    memory_bytes=self.memory_bytes, message_bytes=self.message_bytes, teardown_seconds=self.teardown_seconds,
                    kernel=os.uname().release, machine=os.uname().machine,
                    sources={name: file_digest(Path(path)) for name, path in dict(
                        adapter=__file__, preparation=isolation.__file__, entry=isolation_entry.__file__,
                        bridge=isolation_bridge.__file__, processes=processes.__file__).items()})

    def run(self, source, inputs, *, output, deadline, cancelled=None, handlers=None, max_calls=1):
        from time import monotonic
        if self.identity() != self._frozen:
            raise ValueError("Frozen isolation configuration changed")
        remaining = deadline - monotonic()
        finite_seconds(remaining)
        return isolation.execute(source, inputs, runtime=self.runtime, output=output, timeout_s=remaining,
            deadline=deadline, memory_bytes=self.memory_bytes, output_bytes=self.message_bytes,
            expected_runtime_sha256=self.runtime_sha256, teardown_s=self.teardown_seconds,
            cancelled=cancelled, handlers={} if handlers is None else handlers, max_calls=max_calls)
