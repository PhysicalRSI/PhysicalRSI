"""Frozen, optional analysis helpers proposed by Astra for its shell workspace.

Admission checks names, size, and Python syntax without executing submitted
code. Helpers have no simulator or hardware handle; actions still pass through
the unchanged bounded environment tools. Syntax admission is not a proof that
a helper is correct, useful, or secure; paired execution evaluates usefulness.
"""
import ast
import json
from pathlib import Path
import re

from .common import file_hash, save


def validate_programs(programs):
    if not isinstance(programs,list) or len(programs)>3:
        raise ValueError("Submit at most three analysis helper programs")
    names=set()
    for item in programs:
        if not isinstance(item,dict) or set(item)!={"name","source","usage"}:
            raise ValueError("Each helper needs name, source and usage")
        name,source,usage=item["name"],item["source"],item["usage"]
        if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}\.py",name) or name in names:
            raise ValueError("Use unique lowercase Python basenames")
        if not isinstance(source,str) or not 1<=len(source)<=20000 or not isinstance(usage,str) or not 1<=len(usage)<=2000:
            raise ValueError("Helper source or usage exceeds its bound")
        try:
            ast.parse(source,filename=name)
            compile(source,name,"exec",dont_inherit=True)
        except SyntaxError as error:
            raise ValueError("Helper has invalid Python syntax: "+name) from error
        names.add(name)
    return programs


def freeze_programs(root,programs):
    root=Path(root)
    entries=[]
    files={}
    for item in validate_programs(programs):
        path=root/"programs"/item["name"]
        path.parent.mkdir(exist_ok=True)
        path.write_text(item["source"])
        name=str(path.relative_to(root))
        sha=file_hash(path)
        entries.append(dict(path=name,sha256=sha,usage=item["usage"]))
        files[name]=sha
    save(root/"programs.json",dict(programs=entries,role="analysis helpers; no direct device access"))
    files["programs.json"]=file_hash(root/"programs.json")
    return files


def read_programs(root):
    root=Path(root)
    manifest=json.loads((root/"programs.json").read_text())
    result=[]
    for item in manifest["programs"]:
        path=root/item["path"]
        if not path.resolve().is_relative_to(root.resolve()) or file_hash(path)!=item["sha256"]:
            raise ValueError("Frozen helper changed")
        result.append(dict(name=path.name,source=path.read_text(),usage=item["usage"]))
    return validate_programs(result)
