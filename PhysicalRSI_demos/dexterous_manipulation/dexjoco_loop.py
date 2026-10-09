"""Run bounded, sequential GPT-as-Policy / PhysicalRSI Core improvement rounds.

Every round loads HarnessState's survivor and uses fresh preregistered cases.
The complete round schedule has one immutable evaluator identity. The model
proposes memory; Core owns comparison, selection, lineage, and round ordering.
"""
import argparse
import json
from pathlib import Path

from PhysicalRSI_core.embodiment import Revision
from PhysicalRSI_core.experiments import Budget
from PhysicalRSI_core.infra.trial_quota import TrialQuota
from PhysicalRSI_core.lineage import HarnessState
from PhysicalRSI_core.self_harness import SelfHarness
from PhysicalRSI_core.self_harness.artifacts import verify_harness
from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
from PhysicalRSI_core.self_harness.protocol import CaseProtocol

from .common import digest, file_hash, save
from .dexjoco_campaign import Proposer, Selector, Suite, freeze_candidate
from .evidence_evaluator import MediaExperimentEvaluator as ExperimentEvaluator
from .research_feedback import freeze_imports


class CampaignCases:
    """One frozen schedule, with a separate Core claim for every round."""
    def __init__(self,root,*,tasks,rounds,development_seed,validation_seed,test_seed,seed_stride):
        if type(rounds) is not int or not 1<=rounds<=100:
            raise ValueError("Declare 1..100 rounds")
        if len(set(tasks))!=len(tasks) or not tasks:
            raise ValueError("Declare distinct tasks")
        if seed_stride<=0:
            raise ValueError("Use a positive round seed stride")
        schedule={}
        used=set()
        for index in range(rounds):
            seeds={split:seed+index*seed_stride for split,seed in
                   (("development",development_seed),("validation",validation_seed),("test",test_seed))}
            if any(type(seed) is not int or seed<=2 or seed in used for seed in seeds.values()) or len(set(seeds.values()))!=3:
                raise ValueError("Every round and split needs a distinct non-final seed")
            used.update(seeds.values())
            schedule[f"round-{index+1:04d}"]={split:{task:[dict(task=task,seed=seed,split=split)]
                for task in tasks} for split,seed in seeds.items()}
        self.root=Path(root)
        self.manifest=dict(schema="dexjoco.campaign-cases/v1",rounds=schedule,
                           final_seeds_sealed=[0,1,2],test_role="registered_only")
        path=self.root/"schedule.json"
        if path.exists():
            if json.loads(path.read_text())!=self.manifest:
                raise ValueError("Frozen campaign cases changed")
        else:
            save(path,self.manifest)
        self.protocols={name:CaseProtocol(self.root/name,splits=splits) for name,splits in schedule.items()}

    def identity(self):
        if json.loads((self.root/"schedule.json").read_text())!=self.manifest:
            raise ValueError("Frozen campaign cases changed")
        return dict(schedule_sha256=digest(self.manifest),
                    protocols={name:protocol.identity() for name,protocol in self.protocols.items()})


class CampaignSuite:
    def __init__(self,base,cases,round_name):
        self.base,self.cases,self.round_name=base,cases,round_name
        self.protocol=cases.protocols[round_name]

    def identity(self):
        # The schedule, including round order, is fixed in advance. A round's
        # case choice is an input, not a new implementation of the evaluator.
        return dict(kind="dexjoco-campaign-suite/v1",suite=self.base.identity(),
                    campaign_cases=self.cases.identity())

    def development_cases(self,parent):
        self.cases.identity()
        return self.protocol.cases("development")

    def validation_cases(self,comparison):
        self.cases.identity()
        if comparison["round_id"]!=self.round_name:
            raise ValueError("Comparison belongs to a different scheduled round")
        return self.protocol.claim_validation(comparison)

    def admit(self,candidate):
        self.cases.identity()
        return self.base.admit(candidate)

    def environment(self,task): return self.base.environment(task)
    def policy(self,candidate,task): return self.base.policy(candidate,task)
    def verifier(self,task): return self.base.verifier(task)


def verify_loaded_rounds(root):
    """Audit actual policy workspaces, not merely the selected candidate name."""
    root=Path(root)
    ledger=json.loads((root/"campaign.json").read_text())
    state=HarnessState(root/"lineage")
    rows=[]
    expected=ledger["anchor_revision"]
    for record in ledger["rounds"]:
        directory=root/"rounds"/f"round-{record['index']+1:04d}"
        anchor=json.loads((directory/"parent.json").read_text())
        if anchor["revision"]!=expected or record["parent_revision"]!=expected:
            raise ValueError("Next round did not load the committed survivor")
        parent=state.read(expected)["harness"]
        plan=json.loads((directory/"experiments/development"/parent["id"]/"plan.json").read_text())
        loaded=[]
        for trial in plan["trials"]:
            evidence=directory/"experiments/development"/parent["id"]/"trials"/trial["run_id"]
            trajectory=json.loads((evidence/"trajectory.json").read_text())
            live=Path(trajectory[0]["observation"]["observation_path"]).parents[2]
            path=live/"galbot/loaded_candidate.json"
            receipt=json.loads(path.read_text())
            programs={Path(name).name:sha for name,sha in parent["components"]["control"].items()
                      if name.startswith("programs/")}
            if (receipt["candidate_sha256"]!=verify_harness(parent) or
                    receipt["memory_sha256"]!=parent["components"]["memory_rules"]["memory.md"] or
                    receipt["program_sha256"]!=programs):
                raise ValueError("Executing GPT-as-Policy loaded a different candidate")
            loaded.append(dict(task=trial["task"],receipt=str(path),sha256=file_hash(path)))
        selection=directory/"selection.json"
        rows.append(dict(round=directory.name,parent_revision=expected,parent_id=parent["id"],
                         loaded=loaded,result_revision=record["result_revision"],outcome=record["outcome"],
                         selection=str(selection) if selection.exists() else None))
        expected=record["result_revision"]
    return dict(rounds=rows,verified_next_round_loads=max(0,len(rows)-1),
                qualification=None,description="Artifact inheritance audit; native receipts decide task success")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source",required=True,help="Path to the compatible native DexJoCo checkout")
    parser.add_argument("--tasks",nargs="+",default=["bimanual_photograph","bimanual_assembly"])
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--codex",default="codex")
    parser.add_argument("--login-home")
    parser.add_argument("--max-actions",type=int,default=1500)
    parser.add_argument("--physics-steps",type=int,default=1500)
    parser.add_argument("--seconds",type=float,default=7200.)
    parser.add_argument("--agent-timeout",type=float,default=240.)
    parser.add_argument("--development-seed",type=int,default=3103)
    parser.add_argument("--validation-seed",type=int,default=3203)
    parser.add_argument("--test-seed",type=int,default=3303)
    parser.add_argument("--seed-stride",type=int,default=1000)
    parser.add_argument("--history-round",action="append",default=[],
                        help="Completed non-final comparison to import as development research history")
    args=parser.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    cases=CampaignCases(root/"case_protocol",tasks=args.tasks,rounds=args.rounds,
        development_seed=args.development_seed,validation_seed=args.validation_seed,
        test_seed=args.test_seed,seed_stride=args.seed_stride)
    future_seeds={case["seed"] for splits in cases.manifest["rounds"].values()
                  for tasks in splits.values() for values in tasks.values() for case in values}
    freeze_imports(root,args.history_round,future_seeds)
    bootstrap=root/"candidates/parent/candidate.json"
    parent=json.loads(bootstrap.read_text()) if bootstrap.exists() else freeze_candidate(bootstrap.parent,"parent")
    state=HarnessState(root/"lineage")
    quota=TrialQuota(root/"quota",max_trials=args.rounds*3*len(args.tasks))
    profile=dict(tasks={t:dict(weight=1,episodes=1,score_range=[0,1],maximum_regression=0) for t in args.tasks},
                 minimum_gain=0,tie_tolerance=0)
    def build_loop(directory):
        suite=CampaignSuite(Suite(directory,args),cases,directory.name)
        evaluator=ExperimentEvaluator(suite=suite,diagnostic_source=args.source,budgets={t:Budget(args.max_actions,args.seconds) for t in args.tasks},
            scope="dexjoco-galbot-direct-development",system2=Revision("physicalrsi-core","3c8cf7f12d4041babe30bab98854cc504a5afe88"),
            quota=quota)
        return SelfHarness(directory,state=state,proposer=Proposer(evaluator,root,args),evaluator=evaluator,
            selector=Selector(),profile=profile,protocol=dict(identity=evaluator.revision,evaluation_kind="policy_evaluation"),
            scope="dexjoco-galbot-direct-development",max_candidates=1)
    campaign=ImprovementCampaign(root,state=state,build_loop=build_loop,max_rounds=args.rounds)
    result=campaign.run(parent)
    save(root/"result.json",result)
    save(root/"loaded_rounds.json",verify_loaded_rounds(root))
    print(json.dumps(dict(state=result["status"],completed_rounds=result["completed_rounds"],
                          survivor=result["current"]["harness"]["id"],evidence=str(root))),flush=True)


if __name__=="__main__": main()
