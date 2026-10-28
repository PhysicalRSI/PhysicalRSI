"""Unified CLI for the generic PhysicalRSI research workbench.

The command surface deliberately stays task-agnostic.  A user supplies a
task package and harness, while the package chooses whether its environment is
real or simulated and the Core Self-Harness owns the evolution record.
"""
import argparse
import json
import sys
from pathlib import Path


def _top_level_help():
    print("""usage: physicalrsi {evolve,harness,study,skills} ...

Mainline commands:
  evolve    run or resume a user-defined evolution adapter
  harness   design, import, or show a data-only harness
  study     run a bounded framework study
  skills    inspect persisted skill records

Interactive compatibility surface:
  ./physicalrsi --plain --workspace WORKSPACE --command '/help'

Use 'physicalrsi <command> --help' for command-specific options.""")


def _robodyna_compatibility(arguments):
    """Keep the historical RoboDyna study invocation out of the mainline parser."""
    parser = argparse.ArgumentParser(prog='physicalrsi study')
    parser.add_argument('name', choices=['robodyna'])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture', action='store_true', help='Use the CPU contract fixture')
    mode.add_argument('--native', action='store_true', help='Run the native two-task research study')
    parser.add_argument('--upstream', help='Pinned external RoboDyna checkout')
    parser.add_argument('--python', dest='native_python', help='Isolated simulator Python executable')
    parser.add_argument('--runtime-env', help='JSON mapping of native process environment variables')
    parser.add_argument('--evaluation-mode', choices=['frozen', 'async'], default='frozen')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--joint-after', help='Start joint evolution from a completed native study workspace')
    parser.add_argument('--worker-output', help='Native worker staging directory; Core retains exported evidence')
    args = parser.parse_args(arguments[1:])
    if args.fixture:
        if args.joint_after or args.worker_output:
            parser.error('--joint-after and --worker-output require --native')
        from examples.robodyna.study import run
        result = run(args.workspace)
    else:
        if not args.upstream or not args.native_python:
            parser.error('--native requires --upstream and --python')
        from PhysicalRSI_baselines.robodyna.adapter import NativeSuite
        from examples.robodyna.native_study import run, run_joint
        environment = json.loads(Path(args.runtime_env).read_text()) if args.runtime_env else {}
        suite = NativeSuite(
            args.upstream,
            args.native_python,
            runtime_environment=environment,
            output_root=Path(args.worker_output) if args.worker_output else Path(args.workspace) / 'workers',
        )
        if args.joint_after:
            result = run_joint(
                args.workspace,
                predecessor_root=args.joint_after,
                suite=suite,
                mode=args.evaluation_mode,
            )
        else:
            result = run(
                args.workspace,
                suite=suite,
                parent_parameters=dict(velocity_gain=1., rolling_energy_factor=0., max_horizon_s=30.),
                baseline_parameters=dict(velocity_gain=.5, rolling_energy_factor=0., max_horizon_s=30.),
                development=(101, 102),
                validation=(201, 202, 203),
                heldout=(10000, 10001, 10002),
                mode=args.evaluation_mode,
            )
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] in (["--help"], ["-h"]):
        _top_level_help()
        return 0
    if len(arguments) >= 2 and arguments[:2] == ['study', 'robodyna']:
        return _robodyna_compatibility(arguments)
    if not arguments or arguments[0] not in {'evolve', 'harness', 'study', 'skills'}:
        from PhysicalRSI.cli import main as legacy
        return legacy(arguments)
    parser = argparse.ArgumentParser(prog='physicalrsi')
    commands = parser.add_subparsers(dest='command', required=True)
    evolve = commands.add_parser('evolve', help='Run or resume a user-defined RSI evolution loop')
    evolve.add_argument('--workspace', required=True, type=Path)
    evolve.add_argument('--task', type=Path,
                        help='physicalrsi.task/v1 package supplied by the user')
    evolve.add_argument('--harness', type=Path,
                        help='physicalrsi.harness/v1 descriptor to bind before evolution')
    evolve.add_argument('--scheme', choices=['hybrid', 'learned_skill', 'code_policy'],
                        default=None)
    harness = commands.add_parser('harness', help='Design or import a data-only harness')
    harness.add_argument('action', choices=['show', 'design', 'import'])
    harness.add_argument('--workspace', required=True, type=Path)
    harness.add_argument('--definition', type=Path,
                         help='JSON definition for design, or descriptor for import')
    study = commands.add_parser('study', help='Run a bounded framework study')
    study.add_argument('name', choices=['fixture'])
    study.add_argument('--workspace', required=True, type=Path)
    registry = commands.add_parser('skills', help='Inspect persisted skill descriptors')
    registry.add_argument('--root', required=True)
    args = parser.parse_args(arguments)
    if args.command == 'evolve':
        from PhysicalRSI.application import Application
        application = Application(args.workspace)
        if args.harness is not None:
            imported = json.loads(args.harness.read_text())
            if imported.get('schema') == 'physicalrsi.rsi-plan/v1':
                application.rsi_plan('set', imported)
            else:
                application.harness('import', args.harness)
        if args.scheme is not None:
            plan = application.rsi_plan('show')
            plan['scheme'] = args.scheme
            application.rsi_plan('set', plan)
        if args.task is not None:
            application.load_task(args.task)
        result = application.evolve()
    elif args.command == 'harness':
        from PhysicalRSI.application import Application
        application = Application(args.workspace)
        if args.action == 'show':
            result = application.harness('show')
        elif args.action == 'import':
            if args.definition is None:
                parser.error('harness import requires --definition')
            result = application.harness('import', args.definition)
        else:
            if args.definition is None:
                parser.error('harness design requires --definition')
            definition = json.loads(args.definition.read_text())
            result = application.harness('design', definition)
    elif args.command == 'study':
        from examples.robodyna.study import run
        result = run(args.workspace)
    else:
        from .skills.registry import SkillRegistry
        result = SkillRegistry(args.root).list()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
