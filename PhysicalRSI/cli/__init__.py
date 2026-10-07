"""Minimal physicalRSI command line entry point."""

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "operator":
        from PhysicalRSI.operator import main as operator_main
        return operator_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="physicalrsi", description="physicalRSI · composable embodied intelligence",
        epilog="Use 'physicalrsi operator --help' for explicit local device inspection and recovery.",
    )
    parser.add_argument("--workspace", type=Path, default=Path(".physicalrsi"))
    parser.add_argument("--plain", action="store_true", help="Plain terminal text")
    parser.add_argument("--version", action="version", version="PhysicalRSI 0.2.0")
    parser.add_argument("--port", type=int, help="Local preview port")
    parser.add_argument("action", nargs="?", choices=["preview", "doctor"])
    parser.add_argument("section", nargs="?", choices=["piano", "dexjoco", "baseline"], default="piano")
    parser.add_argument(
        "--command",
        action="append",
        help="Run a slash command; repeat for multiple commands",
    )
    args = parser.parse_args(argv)
    if args.port is not None:os.environ['PHYSICALRSI_SHOWCASE_PORT']=str(args.port)
    if args.action:
        if args.command:parser.error('Use an action or --command, not both')
        args.command=['/doctor' if args.action=='doctor' else
                      ('/baselines' if args.section=='baseline' else '/demo '+args.section)]
    from PhysicalRSI.application import Application

    from .terminal import run_console

    return run_console(Application(args.workspace), args.command, plain=args.plain)
