"""Bind original RGB and videos to Core comparison evidence, as well as JSON."""
import json
from pathlib import Path

from PhysicalRSI_core.self_harness.evaluation import ExperimentEvaluator as CoreExperimentEvaluator

from .common import file_hash
from .evidence import verify_media
from .working_programs import collect_working_programs
from .task_diagnostics import diagnose_trajectory


def trajectory_media(trajectory,root):
    root=Path(root).resolve()
    trace=json.loads(Path(trajectory).read_text())
    verify_media(trace)
    paths={Path(item["path"]) for step in trace for item in step["observation"]["images"]}
    paths.update(Path(step["observation"]["observation_path"]) for step in trace)
    live=Path(trace[0]["observation"]["observation_path"]).parents[2]
    paths.update(Path(p) for p in collect_working_programs(live)["evidence_sha256"])
    notes=live/"galbot/agent/NOTES.md"
    if notes.is_file(): paths.add(notes)
    paths.update((live/"observations").glob("*/codex_previews/*.png"))
    for name in ("worker.json","tools.json","launch.json","PROMPT.md","loaded_candidate.json"):
        artifact=live/"galbot"/name
        if artifact.is_file(): paths.add(artifact)
    manifest=live/"video_manifest.json"
    video=json.loads(manifest.read_text())
    path=Path(video["path"])
    if file_hash(path)!=video["sha256"]:
        raise ValueError("Rollout video changed")
    paths.update((manifest,path,live/"native_transitions.json"))
    result={}
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Rollout media lies outside round evidence")
        result[str(path.resolve().relative_to(root))]=file_hash(path)
    return result


class MediaExperimentEvaluator(CoreExperimentEvaluator):
    def __init__(self, *, diagnostic_source=None, **kwargs):
        self.diagnostic_source=str(Path(diagnostic_source).resolve()) if diagnostic_source else None
        super().__init__(**kwargs)

    def identity(self):
        result=super().identity()
        result["post_episode_diagnostics"]=dict(source=self.diagnostic_source,
            implementation=file_hash(Path(__file__).with_name("task_diagnostics.py")),
            role="System 2 feedback only; native score unchanged")
        return result

    def _diagnose(self,trajectory,output):
        if self.diagnostic_source is None:
            return None,None
        relative=Path(trajectory).relative_to(Path(output))
        target=Path(output)/"diagnostics"/relative.parent/"task_diagnostic.json"
        report=diagnose_trajectory(trajectory,self.diagnostic_source,target)
        return report,str(target.relative_to(Path(output)))

    def development(self,parent,output):
        feedback=super().development(parent,output)
        for row in feedback["episodes"]:
            trajectory=Path(output)/"experiments/development"/parent["id"]/"trials"/row["run_id"]/"trajectory.json"
            feedback["evidence"].update(trajectory_media(trajectory,output))
            report,name=self._diagnose(trajectory,output)
            if report is not None:
                row["diagnostics"]=report["summary"]
                feedback["evidence"][name]=file_hash(Path(output)/name)
        return feedback

    def evaluate(self,candidate,comparison,cohort,output):
        result=super().evaluate(candidate,comparison,cohort,output)
        for row in result["episodes"]:
            names=[name for name in row["evidence_sha256"] if name.endswith("/trajectory.json")]
            if len(names)!=1:
                raise ValueError("Expected one raw trajectory for each trial")
            row["evidence_sha256"].update(trajectory_media(Path(output)/names[0],output))
            report,name=self._diagnose(Path(output)/names[0],output)
            if report is not None:
                row["diagnostics"]=report["summary"]
                row["evidence_sha256"][name]=file_hash(Path(output)/name)
        return result
