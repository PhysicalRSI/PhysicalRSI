"""Read bounded public working-code evidence; never execute it or read RPC reasoning."""
import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_working_programs(live, *, maximum_files=6, maximum_file_bytes=24000,
                             maximum_total_bytes=96000, maximum_commands=12):
    """Inventory episode scratch Python and public command lifecycle records.

    A successful shell exit does not establish correct geometry or task success.
    Commands are recorded as untrusted evidence, never evaluated by this reader.
    Private app-server RPC logs are deliberately outside this interface.
    """
    live=Path(live).resolve()
    scratch=live/"galbot/agent/scratch"
    result=dict(role="unvalidated episode working code; evidence, not instructions",
                programs=[], omissions=[], public_commands=[], evidence_sha256={})
    if min(maximum_files,maximum_file_bytes,maximum_total_bytes,maximum_commands)<0:
        raise ValueError("Evidence limits must be nonnegative")
    if scratch.is_symlink() or (scratch.exists() and not scratch.resolve().is_relative_to(live)):
        result["omissions"].append(dict(path="scratch",reason="untrusted scratch path"))
        return result
    total=0
    for path in sorted(scratch.rglob("*.py")) if scratch.is_dir() else []:
        name=str(path.relative_to(scratch))
        if (path.is_symlink() or not path.resolve().is_relative_to(scratch.resolve()) or
                any(p.is_symlink() for p in path.parents if p.is_relative_to(scratch))):
            result["omissions"].append(dict(path=name,reason="symlink or external path"))
            continue
        if not path.is_file():
            result["omissions"].append(dict(path=name,reason="not a regular file"))
            continue
        size=path.stat().st_size
        if len(result["programs"])>=maximum_files or size>maximum_file_bytes or total+size>maximum_total_bytes:
            result["omissions"].append(dict(path=name,reason="declared evidence size limit",bytes=size))
            continue
        data=path.read_bytes()
        try: source=data.decode("utf-8")
        except UnicodeDecodeError:
            result["omissions"].append(dict(path=name,reason="not UTF-8 Python source"))
            continue
        total+=len(data)
        sha=hashlib.sha256(data).hexdigest()
        result["programs"].append(dict(path=name,source_path=str(path),sha256=sha,source=source,
                                       execution_status="consult public command evidence; writing is not execution"))
        result["evidence_sha256"][str(path)]=sha
    events=live/"galbot/agent_events.jsonl"
    if events.is_file() and not events.is_symlink() and events.resolve().is_relative_to(live):
        # This is the upstream PUBLIC lifecycle log, not rpc_in/rpc_out or model reasoning.
        result["evidence_sha256"][str(events)]=_sha(events)
        commands=[]
        with events.open() as stream:
            for line in stream:
                try: event=json.loads(line)
                except (ValueError,TypeError): continue
                if not isinstance(event,dict): continue
                item=event.get("item",{})
                if not isinstance(item,dict): continue
                if event.get("method")=="item/completed" and item.get("type")=="commandExecution":
                    command=item.get("command")
                    if not isinstance(command,str): continue
                    if not ("python" in command.lower() or any(Path(p["path"]).name in command for p in result["programs"])):
                        continue
                    output=item.get("aggregatedOutput")
                    output=output if isinstance(output,str) else ""
                    # Bound copied text while retaining the hash and original public log.
                    commands.append(dict(item_id=item.get("id"),completed_at_ms=event.get("completedAtMs"),
                        command=command[:4000],command_truncated=len(command)>4000,
                        output=output[:4000],output_truncated=len(output)>4000,
                        exit_code=item.get("exitCode"),status=item.get("status")))
        result["public_commands"]=commands[-maximum_commands:] if maximum_commands else []
        if len(commands)>maximum_commands:
            result["omissions"].append(dict(path="agent_events.jsonl",reason="earlier public commands omitted",
                                            count=len(commands)-maximum_commands))
    result["interpretation"]=("Commands and exit codes show what the agent attempted. They do not prove a helper was "
        "used correctly or improved the native task. Generalize functions and remove episode coordinates, "
        "pixel guesses, seeds and paths before proposing reusable helpers. No automatic adoption.")
    return result
