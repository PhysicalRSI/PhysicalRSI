"""Declared numeric JSON mutation space and atomic candidate publication."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ..infra.storage import canonical, digest, relative_path
from .artifacts import CLOSURE, HarnessSnapshots, harness_identity, verify_harness


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Search parameters and scores must be finite numbers')
    return float(value)


@dataclass(frozen=True)
class SearchField:
    component: str
    file: str
    key: tuple[str, ...]
    lower: float
    upper: float
    precision: int = 3

    def __post_init__(self):
        if (self.component not in CLOSURE-{'foundation','weights'}
                or str(relative_path(self.file)) != self.file
                or not isinstance(self.key, (tuple,list)) or not self.key
                or any(not isinstance(k,str) or not k for k in self.key)
                or type(self.precision) is not int or not 0 <= self.precision <= 9
                or not finite(self.lower) < finite(self.upper)
                or not math.isfinite(self.upper-self.lower)
                or any(round(v,self.precision) != v for v in (self.lower,self.upper))):
            raise ValueError('Declare a bounded mutable numeric JSON field on a finite precision grid')
        object.__setattr__(self,'key',tuple(self.key))

    def identity(self):
        return dict(component=self.component,file=self.file,key=list(self.key),lower=self.lower,
                    upper=self.upper,precision=self.precision)

    def node(self, document):
        node=document
        for key in self.key[:-1]:
            if not isinstance(node,dict) or key not in node:
                raise ValueError('Search field must address an existing JSON object key')
            node=node[key]
        if not isinstance(node,dict) or self.key[-1] not in node:
            raise ValueError('Search field must address an existing JSON object key')
        return node

    def read(self, document):
        value=finite(self.node(document)[self.key[-1]])
        if not self.lower <= value <= self.upper or round(value,self.precision) != value:
            raise ValueError('Candidate parameter is outside the declared precision grid')
        return value

    def assign(self, document, value):
        self.node(document)[self.key[-1]]=value

    def quantize(self, unit):
        return min(self.upper,max(self.lower,round(self.lower+finite(unit)*(self.upper-self.lower),self.precision)))


class ParameterSpace:
    def __init__(self, fields):
        self.fields=tuple(fields)
        if not self.fields or len(self.fields)>32 or any(not isinstance(f,SearchField) for f in self.fields):
            raise ValueError('Declare between one and 32 numeric search fields')
        for i,first in enumerate(self.fields):
            for second in self.fields[i+1:]:
                if first.file != second.file:
                    continue
                n=min(len(first.key),len(second.key))
                if first.component != second.component or first.key[:n]==second.key[:n]:
                    raise ValueError('Search fields cannot overlap or give a file multiple component owners')

    def identity(self):
        return dict(fields=[f.identity() for f in self.fields])

    def inspect(self,candidate):
        verify_harness(candidate)
        documents={}
        for f in self.fields:
            if [k for k,v in candidate['components'].items() if f.file in v] != [f.component]:
                raise ValueError('Each parameter file requires exactly its declared component owner')
            if f.file not in documents:
                raw=(Path(candidate['root'])/f.file).read_bytes()
                if sha256(raw).hexdigest()!=candidate['components'][f.component][f.file]:
                    raise ValueError('Parameter document changed before parsing')
                documents[f.file]=json.loads(raw)
        values=[f.read(documents[f.file]) for f in self.fields]
        fixed=deepcopy(documents)
        for index,f in enumerate(self.fields):
            f.assign(fixed[f.file],{'physicalrsi.search-slot/v1':index})
        components=deepcopy(candidate['components'])
        for f in self.fields:
            components[f.component][f.file]=digest(fixed[f.file])
        return dict(values=values,documents=documents,condition_sha256=digest(harness_identity(dict(candidate,components=components))))

    def unit(self,values):
        if len(values)!=len(self.fields):
            raise ValueError('Search vector dimensions differ from its space')
        return [(v-f.lower)/(f.upper-f.lower) for f,v in zip(self.fields,values)]

    def quantize(self,values):
        if len(values)!=len(self.fields):
            raise ValueError('Search vector dimensions differ from its space')
        return [f.quantize(v) for f,v in zip(self.fields,values)]

    def candidate(self,parent,inspected,values,*,output,name):
        if self.inspect(parent)!=inspected:
            raise ValueError('Parameter inspection does not describe the current parent')
        if len(values)!=len(self.fields):
            raise ValueError('Search vector dimensions differ from its space')
        documents=deepcopy(inspected['documents'])
        components=deepcopy(parent['components'])
        changes={}
        for f,v in zip(self.fields,values):
            f.assign(documents[f.file],v)
            f.read(documents[f.file])
            changes.setdefault(f.component,[]).append(dict(field=f.identity(),value=v))
        payloads={n:canonical(d)+b'\n' for n,d in documents.items()}
        for f in self.fields:
            components[f.component][f.file]=sha256(payloads[f.file]).hexdigest()
        candidate=materialize(parent,components,payloads,output=output,name=name)
        if self.inspect(candidate)['condition_sha256']!=inspected['condition_sha256']:
            raise ValueError('Search candidate changed a fixed controller input')
        return candidate,changes


def materialize(parent,components,payloads,*,output,name):
    destination=Path(output)
    candidate=dict(id=name,root=str(destination),components=components)
    if 'file_modes' in parent:
        candidate['file_modes']=deepcopy(parent['file_modes'])
    if destination.is_symlink():
        raise ValueError('Parameter candidate must be an owned physical directory')
    if not destination.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix='.parameter-',dir=destination.parent))
        try:
            for path in sorted({n for files in components.values() for n in files}):
                target=temporary/relative_path(path)
                target.parent.mkdir(parents=True,exist_ok=True)
                if path in payloads:target.write_bytes(payloads[path])
                else:shutil.copy2(Path(parent['root'])/path,target)
                if 'file_modes' in candidate:target.chmod(candidate['file_modes'][path])
                with target.open('rb') as stream:os.fsync(stream.fileno())
            verify_harness(dict(candidate,root=str(temporary)))
            verify_harness(parent)
            for folder in sorted((p for p in temporary.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                HarnessSnapshots._sync(folder)
            HarnessSnapshots._sync(temporary)
            os.rename(temporary,destination)
            HarnessSnapshots._sync(destination.parent)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    if verify_harness(candidate)!=digest(harness_identity(candidate)):
        raise ValueError('Existing parameter candidate differs from its frozen fit')
    return candidate
