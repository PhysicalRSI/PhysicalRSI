"""One real SelfHarness round: develop, Astra memory proposal, paired evaluation,
native-outcome selection, and durable lineage. No simulator success edits.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.infra.storage import locked
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import CLOSURE, verify_harness
from PhysicalRSI_core.self_harness.protocol import CaseProtocol, PreregisteredSuite
from PhysicalRSI_core.self_harness.selection import select_survivor
from hybrid_rollout.robodojo.io import InputError
from hybrid_rollout.robodojo.skill.run import CodexPolicy
from hybrid_rollout.robodojo.skill.transport import StdioAppServer

from .common import digest, executing_source_manifest, file_hash, implementation_hash, save
from .candidate_programs import freeze_programs, read_programs, validate_programs
from .contracts import object_schema
from .dexjoco_sim import DexJoCoEnvironment, DexJoCoProfile, NativeVerifier, sim_tool_specs
from .policy import GalbotPolicy
from .evidence_evaluator import MediaExperimentEvaluator as ExperimentEvaluator
from .native_dependencies import native_dependencies
from .research_feedback import build_research_feedback, freeze_imports
from .working_programs import collect_working_programs


def freeze_candidate(root, identifier, *, parent=None, memory="No accepted task memory yet.\n", provenance=None, programs=None):
    if programs is None:
        programs=read_programs(parent["root"]) if parent else []
    validate_programs(programs)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).parent
    # The executing source remains read-only during the round. Admission and
    # policy.identity recheck this content closure before and between actions.
    manifest=executing_source_manifest()
    save(root/"source_manifest.json",manifest)
    shutil.copytree(package/"skills/dexjoco-astra-direct",root/"skill")
    (root/"memory.md").write_text(memory)
    declarations=dict(
        foundation=dict(model="gpt-6-astra",provider="openai",effort="xhigh",weights="remote; not modified"),
        skill_selection=dict(skill="dexjoco-astra-direct",mode="direct"),
        tools=sim_tool_specs(),control=dict(source_manifest="source_manifest.json",adapter=implementation_hash()),
        prompts=dict(skill="skill/SKILL.md",memory="memory.md"),
        dependencies=dict(source_manifest="source_manifest.json",python="3.11",simulator="MuJoCo 3.4.0"),
        weights=dict(kind="remote_foundation",local_weights=None),
        assets=dict(source_manifest="source_manifest.json",environment="native DexJoCo; frozen per experiment"),
        configuration=dict(observation="RGB3+proprio46+robot/camera calibration+robot FK",
            task_context="Declared fixed task rules; no scene-object poses",track="online-agent-development"))
    components={}
    for kind,value in declarations.items():
        save(root/(kind+".json"),value)
        components[kind]={kind+".json":file_hash(root/(kind+".json"))}
    components["skills"]={"skill/SKILL.md":file_hash(root/"skill/SKILL.md")}
    components["memory_rules"]={"memory.md":file_hash(root/"memory.md")}
    components["control"]["source_manifest.json"]=file_hash(root/"source_manifest.json")
    components["control"].update(freeze_programs(root,programs))
    candidate=dict(id=identifier,root=str(root.resolve()),components=components)
    if parent:
        candidate.update(parent_sha256=verify_harness(parent),method="Astra reflection on development feedback",
            changes="Frozen task memory and optional analysis helpers; foundation, device adapter, action limits and verifier unchanged",
            environment="DexJoCo simulation",evidence=provenance,costs=dict(candidate_count=1))
    save(root/"candidate.json",candidate)
    verify_harness(candidate)
    return candidate


def verify_executing_source(candidate):
    verify_harness(candidate)
    manifest=json.loads((Path(candidate["root"])/"source_manifest.json").read_text())
    if executing_source_manifest()!=manifest:
        raise ValueError("Executing source differs from the candidate's frozen closure")


class CaseEnvironment(DexJoCoEnvironment):
    def identity(self):
        result=super().identity()
        result["seed"]="frozen_in_experiment_case"
        return result

    def reset(self,case,context):
        self.seed=case["seed"]
        if self.seed in {0,1,2} or self.seed<0:
            raise ValueError("Final seeds are sealed")
        result=super().reset(case,context)
        root=getattr(self,"initial_reference_root",None)
        if root is not None:
            observation=result["observation"]
            fingerprint=digest(dict(state=observation["state"],
                images=[dict(camera=i["camera"],sha256=i["sha256"]) for i in observation["images"]],
                calibration=observation["camera_calibration"]))
            directory=Path(root)/self.task
            reference=directory/(str(self.seed)+".json")
            with locked(directory/".lock"):
                if reference.exists():
                    saved=json.loads(reference.read_text())
                    if saved["initial_sha256"]!=fingerprint:
                        return dict(result,ready=False,reason="Paired reset RGB/proprio/calibration differs",
                                    initial_sha256=fingerprint)
                else:
                    for other in directory.glob("*.json"):
                        if json.loads(other.read_text())["initial_sha256"]==fingerprint:
                            return dict(result,ready=False,reason="Different case seed reused an identical initial scene",
                                        initial_sha256=fingerprint)
                    save(reference,dict(seed=self.seed,initial_sha256=fingerprint,
                        first_policy_revision=context.harness_revision,
                        observation_path=observation["observation_path"]))
            result["initial_sha256"]=fingerprint
        return result


class FrozenPolicy(GalbotPolicy):
    def __init__(self,*args,candidate,**kwargs):
        self.candidate=candidate
        super().__init__(*args,**kwargs)

    def identity(self):
        verify_executing_source(self.candidate)
        return dict(super().identity(),freeze_sha256=verify_harness(self.candidate))

    def describe(self):
        return replace(super().describe(),revision=verify_harness(self.candidate))

    def begin_episode(self,task,case,observation,context):
        return super().begin_episode(self.environment.prompt,case,observation,context)

    def end_episode(self,context):
        try:
            return super().end_episode(context)
        finally:
            self.environment.close()


class Suite:
    def __init__(self,root,args):
        self.root,self.args=Path(root),args
        self.active={}
        self.dependencies=native_dependencies(str(Path(args.source).resolve()))
        self.dependencies.check(hash_bytes=True)
        save(self.root/"native_dependencies.json",self.dependencies.manifest)

    def identity(self):
        return dict(name="galbot-dexjoco-memory-suite",revision=implementation_hash(),
            native_dependencies_sha256=self.dependencies.check(),
            source=str(Path(self.args.source).resolve()),physics_steps=self.args.physics_steps,
            agent_timeout=self.args.agent_timeout,task_scope=self.args.tasks)

    def admit(self,candidate):
        verify_executing_source(candidate)
        self.dependencies.check(hash_bytes=True)
        return dict(accepted=True,evidence={str(Path(candidate["root"])/"source_manifest.json"):
                    file_hash(Path(candidate["root"])/"source_manifest.json"),
                    str(self.root/"native_dependencies.json"):file_hash(self.root/"native_dependencies.json")},
                    reason="Source, skill and candidate memory match frozen inputs")

    def environment(self,task):
        root=self.root/"live"/task/uuid.uuid4().hex
        root.mkdir(parents=True)
        env=CaseEnvironment(self.args.source,task,1103,root,max_physics_steps=self.args.physics_steps)
        env.initial_reference_root=self.root/"initial_references"
        self.active[task]=env
        return env

    def policy(self,candidate,task):
        env=self.active[task]
        profile=DexJoCoProfile(auth="chatgpt",login_home=self.args.login_home,
                              memory=Path(candidate["root"])/"memory.md")
        profile.skill_root=Path(candidate["root"])/"skill"
        profile.candidate=candidate
        return FrozenPolicy(env,profile,env.evidence/"galbot",codex=self.args.codex,
                            timeout=self.args.agent_timeout,candidate=candidate)

    def verifier(self,task): return NativeVerifier(self.active[task])


class ReflectionProfile(DexJoCoProfile):
    def __init__(self,feedback,**kwargs):
        super().__init__(**kwargs)
        self.feedback=feedback
        self.skill_root=Path(__file__).parent/"skills/dexjoco-rsi-reflection"

    def tool_specs(self):
        return [dict(type="function",name="rsi_read_feedback",description="Read completed development evidence.",
                     inputSchema=object_schema({})),
                dict(type="function",name="rsi_submit_candidate",description="Submit one bounded memory and optional analysis-helper candidate.",
                     inputSchema=object_schema(dict(memory_markdown=dict(type="string"),
                         evidence_ids=dict(type="array",items=dict(type="string")),
                         programs=dict(type="array",maxItems=3,items=object_schema(dict(
                             name=dict(type="string"),source=dict(type="string"),usage=dict(type="string")))))))]

    def prepare_workspace(self,audit):
        agent=super().prepare_workspace(audit)
        save(agent/"context/development_feedback.json",self.feedback)
        (agent/"AGENTS.md").write_text("Follow the injected reflection skill. Read only the supplied development feedback.\n")
        return agent

    def initial_prompt(self,rollout):
        return "Read development feedback and parent artifacts with rsi_read_feedback, inspect supplied RGB keyframes as useful, then propose ONE candidate using rsi_submit_candidate."


class ReflectionTools:
    def __init__(self,feedback):
        self.feedback=feedback
        self.phase,self.tick,self.counters="start",0,{}
        self.result=None
    def handlers(self): return dict(rsi_read_feedback=self.read,rsi_submit_candidate=self.submit)
    def next_call(self): return dict(tool="rsi_read_feedback" if self.phase=="start" else "rsi_submit_candidate")
    def read(self):
        self.phase="act"
        return self.feedback
    def submit(self,memory_markdown,evidence_ids,programs=None):
        known={x["id"] for x in self.feedback["episodes"]}
        if self.phase!="act" or not isinstance(memory_markdown,str) or not 100<=len(memory_markdown)<=12000:
            raise InputError("Read feedback and submit 100..12000 characters of grounded memory")
        if not evidence_ids or not set(evidence_ids)<=known:
            raise InputError("Cite actual development episode IDs")
        try:
            validate_programs(programs or [])
        except ValueError as error:
            raise InputError(str(error)) from error
        self.result=dict(memory_markdown=memory_markdown,evidence_ids=evidence_ids,programs=programs or [])
        self.phase="done"
        self.tick=1
        return dict(submitted=True,rollout_finished=True)


class Proposer:
    def __init__(self,evaluator,root,args):
        self.evaluator,self.root,self.args=evaluator,Path(root),args
    def identity(self):
        imports=self.root/"history_imports.json"
        return dict(name="Astra-grounded-memory-proposer",revision=implementation_hash(),
                    history_imports_sha256=file_hash(imports) if imports.exists() else None)
    def develop(self,parent,output):
        feedback=self.evaluator.development(parent,output)
        history=build_research_feedback(self.root,output,source=self.args.source,tasks=self.args.tasks)
        feedback["research_history"]=history
        for path in Path(output).glob("research_*/**/*.json"):
            feedback["evidence"][str(path.relative_to(output))]=file_hash(path)
        for path in Path(output).glob("history_diagnostics/**/*.json"):
            feedback["evidence"][str(path.relative_to(output))]=file_hash(path)
        feedback["evidence"]["research_history.json"]=file_hash(Path(output)/"research_history.json")
        return feedback
    def resume_development(self,parent,output): return self.develop(parent,output)
    def propose(self,parent,feedback,output):
        output=Path(output)
        episodes=[]
        for row in feedback["episodes"]:
            trial=output/"experiments/development"/parent["id"]/"trials"/row["run_id"]
            trace=json.loads((trial/"trajectory.json").read_text())
            observations=[s["observation"] for s in trace]
            evidence_root=Path(observations[0]["observation_path"]).parents[2]
            notes=evidence_root/"galbot/agent/NOTES.md"
            # Original observations only: no reconstructed frames or object-state oracle.
            indices=sorted({round(i*(len(observations)-1)/4) for i in range(5)})
            keyframes=[dict(observation_path=observations[i]["observation_path"],
                           native_control_steps=observations[i]["native_control_steps"],
                           images=observations[i]["images"]) for i in indices]
            episodes.append(dict(id=row["run_id"],task=row["task"],outcome=row["outcome"],
                measurements=row["measurements"],physics_steps=observations[-1]["physics_steps"],
                timing=observations[-1]["timing"],
                actions=[dict(response=s["action"],execution=s["observation"]["execution"],
                              measured_eef=s["observation"]["current_eef"]) for s in trace[1:]],
                keyframes=keyframes,task_context=observations[0]["task_context"],
                diagnostics=row.get("diagnostics"),
                notes=notes.read_text() if notes.exists() else "No notes written",
                working_programs=collect_working_programs(evidence_root)))
        packet=dict(episodes=episodes,parent_memory=(Path(parent["root"])/"memory.md").read_text(),
                    research_history=feedback.get("research_history",{}),
                    parent_programs=read_programs(parent["root"]),
                    parent_sha256=verify_harness(parent),
                    task_instructions="Revise a specific observed bottleneck using working code, diagnostics and prior failed candidates; state a falsifiable change, preserve useful procedures and exclude scene coordinates",
                    diagnostics_role="Checkpoint-matched post-episode simulation diagnosis for System 2; unavailable as an online policy sensor",
                    native_control_hz=50,maximum_chunk_steps=30,validation_seeds_unseen=True)
        save(output/"reflection_input.json",packet)
        profile=ReflectionProfile(packet,auth="chatgpt",login_home=self.args.login_home)
        rollout=ReflectionTools(packet)
        def transport(argv,workspace):
            env=profile.child_environment(workspace)
            return StdioAppServer(argv,workspace,popen=lambda *a,**kw:subprocess.Popen(*a,**kw,env=env))
        controller=CodexPolicy(output/"reflection",self.args.codex,timeout=self.args.agent_timeout,
            method="gpt_only",runtime=profile,transport_factory=transport)
        try: controller.run(rollout)
        finally:
            controller.close()
            profile.cleanup_workspace(output/"reflection")
        if rollout.result is None: raise RuntimeError("Astra did not submit a memory candidate")
        save(output/"memory_proposal.json",rollout.result)
        identifier="child-"+output.name
        candidate=freeze_candidate(self.root/"candidates"/identifier,identifier,parent=parent,
            memory=rollout.result["memory_markdown"],programs=rollout.result["programs"],provenance={str(output/"reflection_input.json"):
            file_hash(output/"reflection_input.json"),str(output/"memory_proposal.json"):file_hash(output/"memory_proposal.json")})
        return [candidate]


class Selector:
    def identity(self): return dict(name="physicalrsi-native-success-selector",revision=implementation_hash())
    def select(self,comparison,cohort,results,*,evidence_root):
        return select_survivor(comparison,cohort,results,evidence_root=evidence_root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source",required=True,help="Path to the compatible native DexJoCo checkout")
    parser.add_argument("--tasks",nargs="+",default=["bimanual_photograph","bimanual_assembly"])
    parser.add_argument("--codex",default="codex")
    parser.add_argument("--login-home")
    parser.add_argument("--max-actions",type=int,default=1500)
    parser.add_argument("--physics-steps",type=int,default=1500)
    parser.add_argument("--seconds",type=float,default=7200.)
    parser.add_argument("--development-seed",type=int,default=2203)
    parser.add_argument("--validation-seed",type=int,default=2303)
    parser.add_argument("--test-seed",type=int,default=2403)
    parser.add_argument("--agent-timeout",type=float,default=240.)
    parser.add_argument("--history-round",action="append",default=[],
                        help="Completed non-final comparison to import as development research history")
    args=parser.parse_args()
    seeds=[args.development_seed,args.validation_seed,args.test_seed]
    if len(set(seeds))!=3 or any(seed<=2 for seed in seeds):
        parser.error("Use three distinct, non-final seeds greater than 2")
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    freeze_imports(root,args.history_round,seeds)
    round_dir=root/"round-01"
    parent=freeze_candidate(root/"candidates/parent","parent")
    # Test cases remain sealed and are not executed by this proof-of-loop round.
    splits={split:{task:[dict(task=task,seed=seed,split=split)] for task in args.tasks}
            for split,seed in (("development",args.development_seed),("validation",args.validation_seed),("test",args.test_seed))}
    protocol=CaseProtocol(root/"case_protocol",splits=splits)
    suite=PreregisteredSuite(Suite(round_dir,args),protocol)
    evaluator=ExperimentEvaluator(suite=suite,diagnostic_source=args.source,budgets={t:Budget(args.max_actions,args.seconds) for t in args.tasks},
        scope="dexjoco-galbot-direct-development",system2=Revision("physicalrsi-core","3c8cf7f12d4041babe30bab98854cc504a5afe88"),
        quota=TrialQuota(root/"quota",max_trials=3*len(args.tasks)))
    profile=dict(tasks={t:dict(weight=1,episodes=1,score_range=[0,1],maximum_regression=0) for t in args.tasks},
                 minimum_gain=0,tie_tolerance=0)
    state=HarnessState(root/"lineage")
    harness=SelfHarness(round_dir,state=state,proposer=Proposer(evaluator,root,args),evaluator=evaluator,
        selector=Selector(),profile=profile,protocol=dict(identity=evaluator.revision,evaluation_kind="policy_evaluation"),
        scope="dexjoco-galbot-direct-development",max_candidates=1)
    state.initialize(parent,policy=harness.config,scope=harness.config["scope"])
    result=harness.run()
    save(root/"result.json",result)
    print(json.dumps(dict(state="completed",survivor=result["harness"]["id"],decision=result.get("decision"),
                          evidence=str(root))),flush=True)


if __name__ == "__main__": main()
