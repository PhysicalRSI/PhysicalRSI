"""Frozen F/H closure, extracted from the reviewed v1 Self-Harness."""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
import tempfile

from PhysicalRSI_core.infra.storage import (
    digest,
    atomic_json,
    file_digest,
    identifier,
    read_json,
    relative_path,
    locked,
)

CLOSURE = {
    "foundation",
    "skill_selection",
    "skills",
    "tools",
    "control",
    "prompts",
    "memory_rules",
    "dependencies",
    "weights",
    "assets",
    "configuration",
}


def harness_identity(h):
    """Declared identity, usable even when physical inputs need repair.

    Historical declarations without file_modes retain their byte-only identity.
    New mode declarations cover every file and bind regular POSIX permission bits.
    """
    components = h['components']
    if (not isinstance(components, dict) or set(components) != CLOSURE
            or any(not isinstance(files, dict) or not files for files in components.values())):
        raise ValueError('Incomplete F/H component declaration')
    value = dict(components=deepcopy(components))
    if 'file_modes' in h:
        modes = h['file_modes']
        names = {n for files in h['components'].values() for n in files}
        if (not isinstance(modes, dict) or set(modes) != names
                or any(type(m) is not int or not 0 <= m <= 0o777 for m in modes.values())):
            raise ValueError('Declare exact file_modes coverage with POSIX permission bits')
        if any(str(relative_path(n)) != n for n in names):
            raise ValueError('Mode-bound file names must be canonical')
        value['file_modes'] = deepcopy(modes)
    return value


def bind_harness_modes(h):
    """Explicitly bind modes of verified source files before freezing a new run.

    Never rebind an existing declaration to hide drift. Ownership, directory
    permissions, ACLs and the operating-system environment are provider inputs.
    """
    verify_harness(h)
    value = deepcopy(h)
    if 'file_modes' not in value:
        value['file_modes'] = {n: stat.S_IMODE((Path(h['root']) / n).stat().st_mode)
                              for files in h['components'].values() for n in files}
    verify_harness(value)
    return value


def verify_harness(h):
    """Verify all declared physical inputs, including referenced asset manifests.

    Closure completeness is independently attested by admission, not inferred
    from file extensions. Large assets may be covered by a pinned manifest, whose
    physical verification must be performed by the dependency_closure gate.
    """
    identifier(h["id"])
    identity = harness_identity(h)
    source_root = Path(h["root"])
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Harness root must be a physical directory")
    root = source_root.resolve()
    if set(h["components"]) != CLOSURE:
        raise ValueError("Incomplete F/H component declaration")
    for kind, entries in h["components"].items():
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Empty harness component: {kind}")
        for name, sha in entries.items():
            declared = root / relative_path(name)
            if any(
                part.is_symlink()
                for part in [declared, *declared.parents]
                if part != root.parent
            ):
                raise ValueError("Harness inputs must be physical files")
            p = declared.resolve()
            if not p.is_relative_to(root) or not p.is_file() or file_digest(p) != sha:
                raise ValueError(f"Changed or missing harness input: {name}")
            if 'file_modes' in identity and stat.S_IMODE(p.stat().st_mode) != identity['file_modes'][name]:
                raise ValueError(f"Changed harness file mode: {name}")
    return digest(identity)


class _HarnessReadAudit:
    """Reuse checked snapshots inside one read, never across execution boundaries.

    This describes bytes already checked during the current audit, not a lock on
    the live filesystem. Each public read must create a fresh instance.
    """

    def __init__(self):
        self.checked = set()

    def verify(self, harness):
        identifier(harness["id"])
        revision = digest(harness_identity(harness))
        key = (str(harness["root"]), revision)
        if key not in self.checked:
            if verify_harness(harness) != revision:
                raise ValueError("Harness declaration changed during audit")
            self.checked.add(key)
        return revision


@dataclass(frozen=True)
class EvolutionSpec:
    """The component mutation boundary, frozen before candidate generation.

    Model/checkpoint updates need a separately reviewed evolver. Runtime memory
    content can be included under memory_rules; research memory has its own CAS.
    """

    mutable: tuple[str, ...] = tuple(sorted(CLOSURE - {"foundation", "weights"}))

    def __post_init__(self):
        values = tuple(self.mutable)
        if (not values or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values) or not set(values) <= CLOSURE
                or {"foundation", "weights"}.intersection(values)):
            raise ValueError("Declare mutable harness components; model state stays fixed")
        object.__setattr__(self, "mutable", tuple(sorted(values)))

    def identity(self):
        return dict(schema="physicalrsi.evolution-spec/v1", mutable=list(self.mutable),
                    fixed=sorted(CLOSURE - set(self.mutable)))

    @classmethod
    def from_identity(cls, value):
        if value.get("schema") == "physicalrsi.checkpoint-evolution/v1":
            from .training import CheckpointEvolutionSpec
            return CheckpointEvolutionSpec.from_identity(value)
        result = cls(tuple(value["mutable"]))
        if result.identity() != value:
            raise ValueError("Invalid evolution specification")
        return result

    def compare(self, parent, child):
        verify_harness(parent)
        verify_harness(child)
        if ('file_modes' in parent) != ('file_modes' in child):
            raise ValueError('Evolution cannot add or drop file_modes; bind before a new lineage')
        changed = {}
        for kind in CLOSURE:
            names = sorted(set(parent['components'][kind]) | set(child['components'][kind]))
            differences = [n for n in names if parent['components'][kind].get(n) != child['components'][kind].get(n)
                           or parent.get('file_modes', {}).get(n) != child.get('file_modes', {}).get(n)]
            if differences:
                changed[kind] = differences
        if not set(changed) <= set(self.mutable):
            raise ValueError("Candidate changed fixed evolution components: "
                             + ", ".join(sorted(set(changed) - set(self.mutable))))
        return dict(sorted(changed.items()))


class HarnessSnapshots:
    """Owned, content-addressed copies of declared files, never hard links.

    Published snapshots are verified on use; capture never overwrites them.
    Explicit restoration quarantines damaged bytes before replacing them. External
    assets represented by manifests still belong to the dependency provider.
    This is local POSIX persistence, not a sandbox or an external asset backup.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

    def capture(self, harness):
        harness = deepcopy(harness)
        revision = verify_harness(harness)
        target = self.root / revision
        captured = dict(harness, root=str(target))
        with locked(self.root / ".snapshots.lock"):
            if not target.exists():
                temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
                try:
                    paths = {name for entries in harness["components"].values() for name in entries}
                    for name in sorted(paths):
                        destination = temporary / relative_path(name)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(Path(harness["root"]) / name, destination)
                        shutil.copymode(Path(harness["root"]) / name, destination)
                        with destination.open("rb") as stream:
                            os.fsync(stream.fileno())
                    if verify_harness(dict(harness, root=str(temporary))) != revision:
                        raise ValueError("Harness changed while taking snapshot")
                    verify_harness(harness)
                    # Flush nested directory entries before publishing the tree.
                    directories = [p for p in temporary.rglob("*") if p.is_dir()]
                    for directory in [*directories, temporary]:
                        self._sync(directory)
                    os.rename(temporary, target)
                    self._sync(self.root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            if verify_harness(captured) != revision:
                raise ValueError("Published harness snapshot changed")
        return captured

    @staticmethod
    def _inventory(root):
        """Describe quarantined bytes without following links or opening devices."""
        if not root.exists() and not root.is_symlink():
            return None
        values, pending = {}, [root]
        while pending:
            path = pending.pop()
            mode = path.lstat().st_mode
            row = dict(mode=mode)
            if stat.S_ISLNK(mode):
                row["link"] = os.readlink(path)
            elif stat.S_ISREG(mode):
                row["sha256"] = file_digest(path)
            elif stat.S_ISDIR(mode):
                pending.extend(sorted(path.iterdir()))
            values[str(path.relative_to(root))] = row
        return digest(values)

    def restore(self, harness, source, *, operation):
        """Explicitly restore original snapshot bytes; preserve damaged inputs.

        The caller owns lineage ancestry/CAS checks. A prepared replacement and
        durable request allow restart across both directory renames. Ordinary
        capture remains strict and never triggers this operation implicitly.
        """
        harness, source, operation = deepcopy(harness), deepcopy(source), deepcopy(operation)
        revision = digest(harness_identity(harness))
        target = self.root / revision
        if Path(harness["root"]) != target or harness_identity(source) != harness_identity(harness):
            raise ValueError("Restore requires an owned snapshot and its exact declared closure")
        source_root = Path(source["root"]).resolve()
        repairs = self.root.parent / "snapshot-repairs"
        if (source_root.is_relative_to(self.root) or source_root.is_relative_to(repairs)
                or self.root.is_relative_to(source_root) or repairs.is_relative_to(source_root)):
            raise ValueError("Restore requires an independent source outside the snapshot and repair stores")
        request = dict(schema="physicalrsi.snapshot-repair-request/v1", operation=operation,
                       harness=harness, source=source, implementation=file_digest(Path(__file__)))
        request_sha = digest(request)
        folder = repairs / request_sha
        staged, quarantine = folder / "prepared", folder / "quarantine"

        def publish(path, value):
            if path.is_symlink():
                raise ValueError("Snapshot repair records must be physical files")
            if path.exists():
                if read_json(path) != value:
                    raise ValueError("Frozen snapshot repair record changed")
            else:
                atomic_json(path, value)

        def valid_target():
            try:
                return verify_harness(harness) == revision
            except (ValueError, OSError):
                return False

        def completion(plan, quarantine_sha):
            return dict(schema="physicalrsi.snapshot-repair/v1", request_sha256=request_sha,
                        snapshot_sha256=revision, snapshot_root=str(target), operation=operation,
                        restored=plan is not None, quarantine=str(quarantine) if quarantine_sha is not None else None,
                        quarantine_sha256=quarantine_sha, qualification=None)

        with locked(self.root / ".snapshots.lock"):
            if repairs.is_symlink() or folder.is_symlink():
                raise ValueError("Snapshot repair records must use physical directories")
            folder.mkdir(parents=True, exist_ok=True)
            self._sync(repairs)
            self._sync(repairs.parent)
            publish(folder / "request.json", request)
            completed = folder / "completed.json"
            plan_path = folder / "prepared.json"
            if completed.is_symlink() or plan_path.is_symlink():
                raise ValueError("Snapshot repair records must be physical files")
            plan = read_json(plan_path) if plan_path.exists() else None
            if plan is not None and (set(plan) != {"request_sha256", "snapshot_sha256", "damaged_sha256"}
                                     or plan.get("request_sha256") != request_sha
                                     or plan.get("snapshot_sha256") != revision):
                raise ValueError("Prepared snapshot repair identity changed")
            if completed.exists():
                value = read_json(completed)
                quarantine_sha = self._inventory(quarantine)
                if (not valid_target() or value != completion(plan, quarantine_sha)
                        or (plan is not None and quarantine_sha != plan["damaged_sha256"])):
                    raise ValueError("Completed snapshot repair evidence changed")
                return value
            if not valid_target():
                if plan is None:
                    if verify_harness(source) != revision:
                        raise ValueError("Restoration source differs from the original snapshot")
                    if staged.exists() or staged.is_symlink():
                        # A crash after preparation but before its receipt may
                        # leave a complete stage; never overwrite unknown bytes.
                        verify_harness(dict(harness, root=str(staged)))
                    else:
                        temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=folder))
                        try:
                            for name in sorted({name for entries in harness["components"].values() for name in entries}):
                                destination = temporary / relative_path(name)
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(source_root / name, destination)
                                shutil.copymode(source_root / name, destination)
                                with destination.open("rb") as stream:
                                    os.fsync(stream.fileno())
                            verify_harness(dict(harness, root=str(temporary)))
                            verify_harness(source)
                            for directory in [*(p for p in temporary.rglob("*") if p.is_dir()), temporary]:
                                self._sync(directory)
                            os.rename(temporary, staged)
                            self._sync(folder)
                        finally:
                            if temporary.exists():
                                shutil.rmtree(temporary)
                    plan = dict(request_sha256=request_sha, snapshot_sha256=revision,
                                damaged_sha256=self._inventory(target))
                    publish(plan_path, plan)
                verify_harness(dict(harness, root=str(staged)))
                if quarantine.exists() or quarantine.is_symlink():
                    if self._inventory(quarantine) != plan["damaged_sha256"] or target.exists() or target.is_symlink():
                        raise ValueError("Quarantined snapshot or repair destination changed; reconcile before retry")
                else:
                    if self._inventory(target) != plan["damaged_sha256"]:
                        raise ValueError("Damaged snapshot changed before quarantine")
                    if target.exists() or target.is_symlink():
                        os.rename(target, quarantine)
                        self._sync(self.root)
                        self._sync(folder)
                os.rename(staged, target)
                self._sync(self.root)
                self._sync(folder)
            if verify_harness(harness) != revision:
                raise ValueError("Restored snapshot differs from its declared revision")
            quarantine_sha = self._inventory(quarantine)
            if quarantine_sha is not None and (plan is None or quarantine_sha != plan["damaged_sha256"]):
                raise ValueError("Quarantined snapshot changed before completion")
            if plan is not None and quarantine_sha != plan["damaged_sha256"]:
                raise ValueError("Snapshot quarantine is incomplete")
            self._sync(self.root)
            self._sync(folder)
            value = completion(plan, quarantine_sha)
            publish(completed, value)
            return value

    @staticmethod
    def _sync(directory):
        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
