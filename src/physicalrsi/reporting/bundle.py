"""Immutable, hashed report artifacts; only actual environment videos are copied."""
from pathlib import Path
from PhysicalRSI_core.infra.storage import canonical, file_digest, locked, read_json, relative_path


class ReportBundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        path = self.root/relative_path(name)
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError('Report path escapes bundle')
        return path

    def _write(self, name, payload):
        with locked(self.root/'.lock'):
            path = self._path(name)
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError('Report artifact already exists with different content')
                return path
            if (self.root/'manifest.json').exists():
                raise ValueError('Report bundle is finalized')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path

    def write_json(self, name, value):
        return self._write(name, canonical(value)+b'\n')

    def write_text(self, name, value):
        return self._write(name, str(value).encode())

    def add_file(self, source, name=None):
        source = Path(source)
        return self._write(name or source.name, source.read_bytes())

    def finalize(self, *, title, summary, qualification=None):
        self.write_text('report.md', f'# {title}\n\n{summary}\n\nQualification: {qualification!r}\n')
        rows = {str(p.relative_to(self.root)):file_digest(p) for p in sorted(self.root.rglob('*'))
                if p.is_file() and p.name not in {'.lock','manifest.json'}}
        manifest = dict(schema='physicalrsi.report-bundle/v1', artifacts=rows, qualification=qualification)
        self.write_json('manifest.json', manifest)
        return self.verify()

    def verify(self):
        manifest = read_json(self.root/'manifest.json')
        for name, sha in manifest['artifacts'].items():
            if file_digest(self._path(name)) != sha:
                raise ValueError('Report artifact changed')
        return manifest
