import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description='Self-built Isaac Sim lab: sample transfer and evidence-driven repair')
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--isaac-python',default=os.environ.get('PHYSICALRSI_SIM_PYTHON'))
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--core-research',action='store_true',help='Use common Trial/Evaluation and recursive native research')
    p.add_argument('--staged',action='store_true',help='Screen a repair and two negative controls on native development rollouts')
    p.add_argument('--feedback-search',action='store_true',help='Refine carrying height from verified development failures, with native screening')
    p.add_argument('--screen-incumbent',action='store_true',help='Also measure the retained incumbent during development screening')
    p.add_argument('--rounds',type=int,default=2,help='Bounded research rounds with fresh paired validation')
    p.add_argument('--device-registry',type=Path,help='Shared local device registry for all cooperating Isaac clients')
    p.add_argument('--resource',default='isaac-gpu-0',help='Leased GPU resource name; does not select a CUDA device')
    args=p.parse_args(argv)
    if args.staged and not args.core_research:p.error('--staged requires --core-research')
    if args.feedback_search and not args.core_research:p.error('--feedback-search requires --core-research')
    if args.screen_incumbent and not (args.staged or args.feedback_search):p.error('--screen-incumbent requires --staged or --feedback-search')
    if args.report_only:
        if args.core_research:p.error('Research reports are in result.json and campaign/final-test/report.json')
        from .report import render
        print(render(args.workspace));return 0
    if not args.isaac_python:p.error('Pass --isaac-python or set PHYSICALRSI_SIM_PYTHON (Isaac Sim 5.1)')
    if args.core_research:
        from .research import run
        result=run(args.workspace,args.isaac_python,episodes=args.episodes,rounds=args.rounds,gui=args.gui,
                   device_registry=args.device_registry,resource=args.resource,staged=args.staged,
                   screen_incumbent=args.screen_incumbent,feedback_search=args.feedback_search)
        print(json.dumps(result,indent=2));return 0
    from .harness import run
    print(json.dumps(run(args.workspace,args.isaac_python,args.episodes,args.gui),indent=2))
    print('Open '+str(args.workspace.resolve()/'index.html'))
    return 0

if __name__=='__main__':raise SystemExit(main())
