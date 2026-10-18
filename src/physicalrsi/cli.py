"""Unified CLI: study/registry commands plus the unchanged historical CLI."""
import argparse
import json
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {'study', 'skills'}:
        from PhysicalRSI.cli import main as legacy
        return legacy(arguments)
    parser = argparse.ArgumentParser(prog='physicalrsi')
    commands = parser.add_subparsers(dest='command', required=True)
    study = commands.add_parser('study', help='Run or resume a bounded research study')
    study.add_argument('name', choices=['robodyna'])
    mode = study.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture', action='store_true', help='Use the CPU contract fixture')
    mode.add_argument('--native', action='store_true', help='Run the native two-task research study')
    study.add_argument('--upstream', help='Pinned external RoboDyna checkout')
    study.add_argument('--python', dest='native_python', help='Isolated simulator Python executable')
    study.add_argument('--runtime-env', help='JSON mapping of native process environment variables')
    study.add_argument('--evaluation-mode', choices=['frozen', 'async'], default='frozen')
    study.add_argument('--workspace', required=True)
    study.add_argument('--joint-after', help='Start joint evolution from a completed native study workspace')
    study.add_argument('--worker-output', help='Native worker staging directory; Core retains exported evidence')
    registry = commands.add_parser('skills', help='Inspect persisted skill descriptors')
    registry.add_argument('--root', required=True)
    args = parser.parse_args(arguments)
    if args.command == 'study':
        if args.fixture:
            if args.joint_after or args.worker_output:
                parser.error('--joint-after and --worker-output require --native')
            from examples.robodyna.study import run
            result = run(args.workspace)
        else:
            if not args.upstream or not args.native_python:
                parser.error('--native requires --upstream and --python')
            from pathlib import Path
            from PhysicalRSI_baselines.robodyna.adapter import NativeSuite
            from examples.robodyna.native_study import run, run_joint
            environment = json.loads(Path(args.runtime_env).read_text()) if args.runtime_env else {}
            suite = NativeSuite(args.upstream, args.native_python, runtime_environment=environment,
                                output_root=Path(args.worker_output) if args.worker_output else Path(args.workspace)/'workers')
            if args.joint_after:
                result = run_joint(args.workspace, predecessor_root=args.joint_after, suite=suite,
                                   mode=args.evaluation_mode)
            else:
                result = run(args.workspace, suite=suite,
                    parent_parameters=dict(velocity_gain=1., rolling_energy_factor=0., max_horizon_s=30.),
                    baseline_parameters=dict(velocity_gain=.5, rolling_energy_factor=0., max_horizon_s=30.),
                    development=(101, 102), validation=(201, 202, 203),
                    heldout=(10000, 10001, 10002), mode=args.evaluation_mode)
    else:
        from .skills.registry import SkillRegistry
        result = SkillRegistry(args.root).list()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
