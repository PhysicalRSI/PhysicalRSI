"""Original PhysicalRSI code driver and RPC bridge in an unprivileged jail."""
import ast
import ctypes
import ctypes.util
import errno
import math
import os
from pathlib import Path
import shutil
import sys

from .contracts import digest
from .evidence import write_new
from .references import RSI, use_physicalrsi


def validate_source(source):
    use_physicalrsi()
    from PhysicalRSI_baselines.robodojo.skill_proposal import validate_policy_source
    tree = validate_policy_source(source)
    entry = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy"]
    if len(entry) != 1 or [a.arg for a in entry[0].args.args] != ["robot", "memory"]:
        raise ValueError("Exactly one policy(robot, memory) entry is required")
    # This is an admission hint, not proof of generalization. Held-out evaluation
    # must test the same code on changed arrangements and object positions.
    tainted = {"memory"}
    def depends_on_memory(node):
        return any(isinstance(n, ast.Name) and n.id in tainted for n in ast.walk(node))
    changed = True
    while changed:
        before = set(tainted)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and depends_on_memory(node.value):
                for target in node.targets:
                    tainted.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            if isinstance(node, ast.For) and depends_on_memory(node.iter):
                tainted.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
        changed = tainted != before
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"move_cm", "move_delta", "set_joint_target", "replay"}):
            raise ValueError("L1 must use reusable scene-conditioned skills, not fixed motion replay")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "approach" and
                any(depends_on_memory(arg) for arg in node.args[1:] + [k.value for k in node.keywords])):
            raise ValueError("Metric approach targets cannot come from stored memory waypoints")
    return tree


def prepare_runtime(directory):
    use_physicalrsi()
    from PhysicalRSI_core.infra.isolation import build_runtime
    return build_runtime(directory)


def _filter(path):
    """Use the syscall allowlist from the local reference, not an unrestricted exec."""
    tree = ast.parse((RSI / "PhysicalRSI_core/infra/isolation_entry.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "allowed" for t in n.targets))
    names = ast.literal_eval(assignment.value.func.value).split()
    # Python 3.10 on this host implements time.sleep with select; upstream
    # targets 3.11+, whose sleep uses clock_nanosleep. Neither opens a device.
    names.extend(["select", "pselect6"])
    library = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = library.seccomp_init(0x50000 | errno.EPERM)
    if not ctx:
        raise RuntimeError("Cannot create policy syscall filter")
    try:
        for name in names:
            nr = library.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0 and library.seccomp_rule_add(ctx, 0x7FFF0000, nr, 0):
                raise RuntimeError("Cannot configure policy syscall filter")
        # Python's script fopen uses FIOCLEX; arbitrary device ioctls are denied.
        import termios
        class Comparison(ctypes.Structure):
            _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                        ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]
        library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                   ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Comparison)]
        condition = Comparison(1, 4, termios.FIOCLEX, 0)
        if library.seccomp_rule_add_array(ctx, 0x7FFF0000,
                library.seccomp_syscall_resolve_name(b"ioctl"), 1, ctypes.byref(condition)):
            raise RuntimeError("Cannot configure policy ioctl filter")
        with Path(path).open("xb") as stream:
            if library.seccomp_export_bpf(ctx, stream.fileno()):
                raise RuntimeError("Cannot export policy syscall filter")
    finally:
        library.seccomp_release(ctx)


class PolicyRuntime:
    def __init__(self, runtime, *, timeout_s=600, max_calls=10000):
        self.runtime = Path(runtime).resolve()
        if not math.isfinite(timeout_s) or timeout_s <= 0 or type(max_calls) is not int or max_calls <= 0:
            raise ValueError("Finite runtime budget required")
        self.timeout_s, self.max_calls = timeout_s, max_calls

    def execute(self, source, memory, *, handlers, output, cancelled=None):
        validate_source(source)
        use_physicalrsi()
        from PhysicalRSI_baselines.robodojo.code_execution import _DRIVER
        from PhysicalRSI_core.infra.storage import read_json, file_digest
        from PhysicalRSI_core.infra.batch import ProcessJob
        from PhysicalRSI_core.infra.isolation_bridge import run
        if not shutil.which("bwrap"):
            raise RuntimeError("bubblewrap is required; no unrestricted execution fallback")
        manifest = read_json(self.runtime / "manifest.json")
        if manifest.get("schema") != "physicalrsi.isolation-runtime/v1":
            raise ValueError("Invalid Python runtime manifest")
        root = Path(output).resolve()
        root.mkdir(parents=True, exist_ok=False)
        jail = root / "root"
        jail.mkdir()
        reserved = {"policy.py", "input.json", "response.json", "output"}
        for name, sha in manifest["files"].items():
            p = Path(name)
            if not p.parts or p.is_absolute() or ".." in p.parts or p.parts[0] in reserved:
                raise ValueError("Runtime manifest path escaped")
            source_file = self.runtime / p
            if source_file.is_symlink() or not source_file.resolve().is_relative_to(self.runtime):
                raise ValueError("Runtime manifest symlink escaped")
            if file_digest(source_file) != sha:
                raise ValueError("Frozen runtime was modified")
            target = jail / p
            target.parent.mkdir(parents=True, exist_ok=True)
            # Independent copy: an adversarial runtime mutation cannot change the
            # jail through a shared inode after verification.
            shutil.copyfile(source_file, target)
            target.chmod(0o555)
        executable = manifest["executable"]
        if not executable.startswith("/") or executable[1:] not in manifest["files"]:
            raise ValueError("Runtime interpreter is not pinned")
        (jail / "policy.py").write_text(_DRIVER)
        write_new(jail / "input.json", dict(source=source, memory=memory))
        write_new(jail / "response.json", {})
        (jail / "output").mkdir()
        (jail / "dev").mkdir()
        mailbox = jail / "output/result.json"
        mailbox.touch()
        filter_path = root / "seccomp.bpf"
        _filter(filter_path)
        write_new(root / "request.json", dict(source_sha256=digest(source), memory_sha256=digest(memory),
            reference_driver_sha256=digest(_DRIVER), runtime_sha256=digest(manifest),
            capabilities=sorted(handlers), timeout_s=self.timeout_s, max_calls=self.max_calls))
        command = (sys.executable, "-I", "-S", "-B", str(Path(__file__).with_name("_jail.py")),
                   str(jail), executable, str(filter_path), str(math.ceil(self.timeout_s)))
        job = ProcessJob("code-policy", command, root, timeout_s=self.timeout_s)
        result = run(job, root / "process", mailbox, jail / "response.json", handlers,
                     max_calls=self.max_calls, message_bytes=4*1024*1024, cancelled=cancelled)
        write_new(root / "result.json", dict(value=result, independent_success=False))
        return result
