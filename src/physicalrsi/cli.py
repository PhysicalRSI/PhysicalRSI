"""Unified CLI for the generic PhysicalRSI research workbench.

The command surface deliberately stays task-agnostic.  A user supplies a
task package and harness, while the package chooses whether its environment is
real or simulated and the Core Self-Harness owns the evolution record.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
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
