"""Trusted launcher; only the prepared stdlib jail is visible to generated code."""
import os
from pathlib import Path
import resource
import shutil
import sys


def main():
    jail, executable, filter_file, seconds = sys.argv[1:]
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap unavailable")
    fd = os.open(filter_file, os.O_RDONLY)
    os.set_inheritable(fd, True)
    for kind, limit in ((resource.RLIMIT_CPU, int(seconds)),
                        (resource.RLIMIT_AS, 512*1024*1024),
                        (resource.RLIMIT_FSIZE, 8*1024*1024),
                        (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (limit, limit))
    command = [bwrap, "--unshare-all", "--as-pid-1", "--die-with-parent", "--new-session",
               "--ro-bind", jail, "/", "--dev", "/dev", "--chdir", "/",
               "--bind", str(Path(jail) / "output/result.json"), "/output/result.json",
               "--seccomp", str(fd), "--", executable, "-I", "-S", "-B", "/policy.py"]
    os.execve(bwrap, command, dict(LANG="C.UTF-8", PYTHONHASHSEED="0",
              LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/lib64:/lib"))


if __name__ == "__main__":
    main()
