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
    study.add_argument('--fixture', action='store_true', required=True,
                       help='Use software contract fixture; native RoboDyna is not integrated')
    study.add_argument('--workspace', required=True)
    registry = commands.add_parser('skills', help='Inspect persisted skill descriptors')
    registry.add_argument('--root', required=True)
    args = parser.parse_args(arguments)
    if args.command == 'study':
        from examples.robodyna.study import run
        result = run(args.workspace)
    else:
        from .skills.registry import SkillRegistry
        result = SkillRegistry(args.root).list()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
