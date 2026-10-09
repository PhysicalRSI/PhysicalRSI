"""Freeze the complete local DexJoCo package and declared task configuration.

Hash bytes on admission and teardown; check file inventory and stat identities
between actions. This detects ordinary concurrent source/asset edits, not a
malicious host that can replace files and manipulate filesystem metadata.
"""
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from .common import digest, file_hash


class NativeDependencyClosure:
    def __init__(self,source):
        self.source=Path(source).resolve()
        self.files=self._files()
        if not self.files:
            raise ValueError("DexJoCo package has no source/assets to freeze")
        versions={}
        for name in ("mujoco","numpy","scipy","gymnasium","Pillow","PyYAML"):
            try: versions[name]=version(name)
            except PackageNotFoundError: versions[name]=None
        self.manifest=dict(schema="dexjoco.native-dependencies/v1",python=sys.version,
            packages=versions,files={str(p.relative_to(self.source)):file_hash(p) for p in self.files},
            scope="entire local DexJoCo Python package, meshes, scenes, textures and rand_obj task configs")
        self.stamps=self._stamps(self.files)
        self.revision=digest(self.manifest)

    def _files(self):
        package=self.source/"dexjoco/dexjoco"
        paths=[p for p in package.rglob("*") if p.is_file() and
               not any(part in {"__pycache__",".pytest_cache"} for part in p.parts)]
        paths.extend((self.source/"configs/rand_obj").glob("*.yaml"))
        if any(p.is_symlink() for p in paths):
            raise ValueError("Resolve native dependency symlinks explicitly before freezing")
        return sorted(paths)

    @staticmethod
    def _stamps(paths):
        result={}
        for p in paths:
            s=p.stat()
            result[str(p)]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        return result

    def check(self,*,hash_bytes=False):
        files=self._files()
        if files!=self.files or self._stamps(files)!=self.stamps:
            raise ValueError("Native source or asset inventory changed during the frozen experiment")
        if hash_bytes and any(file_hash(self.source/name)!=sha for name,sha in self.manifest["files"].items()):
            raise ValueError("Native dependency bytes changed")
        return self.revision


@lru_cache(maxsize=8)
def native_dependencies(source):
    return NativeDependencyClosure(source)
