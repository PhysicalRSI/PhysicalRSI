import importlib.util
import json
from pathlib import Path

import pytest

from PhysicalRSI_demos.dexterous_manipulation.working_programs import collect_working_programs as collect


def test_source_is_evidence_without_execution_or_private_reasoning(tmp_path):
    scratch=tmp_path/"galbot/agent/scratch"
    scratch.mkdir(parents=True)
    sentinel=tmp_path/"must-not-execute"
    source=f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
    (scratch/"geometry.py").write_text(source)
    (tmp_path/"galbot/rpc_in.jsonl").write_text("PRIVATE_REASONING_MUST_NOT_BE_READ")
    public=tmp_path/"galbot/agent_events.jsonl"
    public.write_text(json.dumps(dict(method="item/completed",completedAtMs=2,item=dict(type="commandExecution",
        id="command1",command="python scratch/geometry.py",exitCode=1,status="failed",aggregatedOutput="failed geometry calculation")))+"\n")
    data=collect(tmp_path)
    assert not sentinel.exists()
    assert data["programs"][0]["source"]==source
    assert data["public_commands"][0]["exit_code"]==1
    assert data["public_commands"][0]["output"]=="failed geometry calculation"
    assert "PRIVATE_REASONING" not in json.dumps(data)
    assert len(data["evidence_sha256"])==2


def test_symlink_and_limits_are_explicit_omissions(tmp_path):
    scratch=tmp_path/"episode/galbot/agent/scratch"
    scratch.mkdir(parents=True)
    outside=tmp_path/"secret.py"
    outside.write_text("PRIVATE_EXTERNAL_CONTENT")
    (scratch/"a_link.py").symlink_to(outside)
    (scratch/"b_large.py").write_text("x"*101)
    (scratch/"c_ok.py").write_text("value=1\n")
    (scratch/"d_extra.py").write_text("value=2\n")
    data=collect(tmp_path/"episode",maximum_files=1,maximum_file_bytes=100)
    assert [x["path"] for x in data["programs"]]==["c_ok.py"]
    assert len(data["omissions"])==3
    assert "PRIVATE_EXTERNAL_CONTENT" not in json.dumps(data)


def test_public_log_filters_types_and_records_truncation(tmp_path):
    folder=tmp_path/"galbot"
    folder.mkdir()
    rows=[dict(method="item/completed",item=dict(type="reasoning",text="PRIVATE"))]
    rows.extend(dict(method="item/completed",completedAtMs=i,item=dict(type="commandExecution",command="python "+"x"*8001,
                     exitCode=0,aggregatedOutput="y"*5000)) for i in range(3))
    (folder/"agent_events.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    data=collect(tmp_path,maximum_commands=1)
    assert len(data["public_commands"])==1 and data["public_commands"][0]["completed_at_ms"]==2
    assert data["public_commands"][0]["command_truncated"]
    assert data["public_commands"][0]["output_truncated"]
    assert data["omissions"][0]["count"]==2
    assert "PRIVATE" not in json.dumps(data)


def test_scratch_root_cannot_alias_another_episode(tmp_path):
    outside=tmp_path/"other"
    outside.mkdir()
    (outside/"geometry.py").write_text("private=1\n")
    parent=tmp_path/"episode/galbot/agent"
    parent.mkdir(parents=True)
    (parent/"scratch").symlink_to(outside,target_is_directory=True)
    data=collect(tmp_path/"episode")
    assert not data["programs"] and data["omissions"]
