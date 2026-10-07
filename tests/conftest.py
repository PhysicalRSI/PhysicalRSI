import ctypes.util
import os
import sys

import pytest


@pytest.fixture(scope="session")
def isolation_runtime(tmp_path_factory):
    if sys.platform != "linux" or os.geteuid() != 0 or not ctypes.util.find_library("seccomp"):
        pytest.skip("Linux chroot privilege and libseccomp are required for actual isolation tests")
    from pathlib import Path
    status = Path("/proc/self/status").read_text()
    capabilities = int(next(line.split()[1] for line in status.splitlines() if line.startswith("CapEff:")), 16)
    if not capabilities & (1 << 18):
        pytest.skip("CAP_SYS_CHROOT is required for actual isolation tests")
    from PhysicalRSI_core.infra.isolation import build_runtime
    root = tmp_path_factory.mktemp("python-isolation") / "runtime"
    build_runtime(root)
    return root
