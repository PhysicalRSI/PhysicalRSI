"""Immutable JSON metadata and owned file trees on a cooperating POSIX store."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .storage import atomic_json, digest, file_digest, locked, read_json, relative_path


def physical(root, name):
    path = root / relative_path(name)
    if str(path.relative_to(root)) != name:
        raise ValueError("Artifact path must be canonical")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)) or not path.is_file():
        raise ValueError("Artifact must be a physical file: " + name)
    return path


def sync(directory):
    descriptor = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_names(root):
    """Enumerate once per directory, without following links or statting twice."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Owned artifact files must be a physical directory")
    pending, names = [root], set()
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise ValueError("Owned artifact must contain physical files")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    names.add(str(path.relative_to(root)))
                else:
                    raise ValueError("Owned artifact contains a non-regular file")
    return names


class OwnedFiles:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def folder(self, revision):
        if not isinstance(revision, str) or len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Expected a content-addressed artifact revision")
        return self.root / revision

    def read(self, revision):
        root = self.folder(revision)
        record = read_json(physical(root, "record.json"))
        version = record.get('schema')
        if digest(record) != revision or version not in {'physicalrsi.owned-files/v1', 'physicalrsi.owned-files/v2'}:
            raise ValueError("Owned artifact record changed")
        expected = {'schema', 'metadata', 'files'} | ({'file_modes'} if version.endswith('/v2') else set())
        if set(record) != expected or not isinstance(record['files'], dict) or not record['files']:
            raise ValueError('Owned artifact declaration fields changed')
        if version.endswith('/v2') and (not isinstance(record['file_modes'], dict)
                or set(record['file_modes']) != set(record['files'])
                or any(type(m) is not int or not 0 <= m <= 0o777 for m in record['file_modes'].values())):
            raise ValueError('Owned artifact file mode declaration changed')
        files = root / "files"
        actual = _file_names(files)
        if actual != set(record["files"]):
            raise ValueError("Owned artifact closure changed")
        for name, sha in record["files"].items():
            path = physical(files, name)
            if file_digest(path) != sha:
                raise ValueError("Owned artifact file changed: " + name)
            if version.endswith('/v2') and stat.S_IMODE(path.stat().st_mode) != record['file_modes'][name]:
                raise ValueError('Owned artifact file mode changed: ' + name)
        return deepcopy(record)

    def capture(self, metadata, files, *, bind_modes=True):
        """Pin modes by default; bind_modes=False only reconstructs legacy v1 records."""
        if not isinstance(files, dict) or not files:
            raise ValueError("Declare a nonempty owned artifact closure")
        files = deepcopy(files)
        for name, entry in files.items():
            if str(relative_path(name)) != name or set(entry) != {"path", "sha256"}:
                raise ValueError("Invalid artifact source declaration")
        if type(bind_modes) is not bool:
            raise ValueError('Declare whether to bind artifact modes')
        body = dict(schema="physicalrsi.owned-files/v2" if bind_modes else "physicalrsi.owned-files/v1", metadata=deepcopy(metadata),
                    files={name: entry["sha256"] for name, entry in files.items()})
        if bind_modes:
            modes = {}
            for name, entry in files.items():
                source = Path(entry['path'])
                if source.is_symlink() or not source.is_file():
                    raise ValueError('Artifact source must be a physical file')
                modes[name] = stat.S_IMODE(source.stat().st_mode)
                if modes[name] > 0o777:
                    raise ValueError('Owned files require regular POSIX permission bits')
            body['file_modes'] = modes
        revision = digest(body)
        target = self.folder(revision)
        with locked(self.root / ".lock"):
            if not target.exists():
                temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
                try:
                    for name, entry in sorted(files.items()):
                        source = Path(entry["path"])
                        if source.is_symlink() or not source.is_file():
                            raise ValueError("Artifact source must be a physical file")
                        destination = temporary / "files" / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.open("rb") as incoming, destination.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                            outgoing.flush()
                            if bind_modes:
                                os.fchmod(outgoing.fileno(), body['file_modes'][name])
                            os.fsync(outgoing.fileno())
                        if file_digest(destination) != entry["sha256"]:
                            raise ValueError("Artifact source changed during capture")
                        if bind_modes and (stat.S_IMODE(source.stat().st_mode) != body['file_modes'][name]
                                or file_digest(source) != entry['sha256']):
                            raise ValueError('Artifact source mode or bytes changed during capture')
                    atomic_json(temporary / "record.json", body)
                    for folder in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                                         key=lambda p: len(p.parts), reverse=True):
                        sync(folder)
                    sync(temporary)
                    os.rename(temporary, target)
                    sync(self.root)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            if self.read(revision) != body:
                raise ValueError("Conflicting owned artifact")
        return revision

    def evidence(self, revision):
        value = self.read(revision)
        root = self.folder(revision)
        return {str(root / "record.json"): file_digest(root / "record.json"),
                **{str(root / "files" / name): sha for name, sha in value["files"].items()}}
