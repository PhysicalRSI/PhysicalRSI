"""Run a bounded, resumable liquid-handling autoresearch study."""
import argparse
import json
from pathlib import Path

from .study import OpentronsSimulator, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--sim-python', type=Path, required=True)
    parser.add_argument('--proposals', type=Path, help='JSON list of hypotheses and bounded policy settings')
    parser.add_argument('--resume', action='store_true', help='Resume an unchanged study; verify recorded evidence first')
    parser.add_argument('--max-rounds', type=int, help='Maximum new development rounds in this invocation')
    parser.add_argument('--trial-seconds', type=float, default=120, help='Timeout per simulator invocation')
    args = parser.parse_args()
    proposals = json.loads(args.proposals.read_text()) if args.proposals else None
    result = run_study(args.workspace, OpentronsSimulator(args.sim_python), proposals=proposals,
                       resume=args.resume, max_rounds=args.max_rounds, trial_seconds=args.trial_seconds)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] != 'completed' or result['validation_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
