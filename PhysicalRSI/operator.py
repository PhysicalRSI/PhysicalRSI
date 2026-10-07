"""Dedicated local operator CLI, outside conversation/policy tool dispatch."""

import argparse
import json
from pathlib import Path
import sys

from PhysicalRSI_core.infra.operator import OperatorClient, recovery_request
from PhysicalRSI_core.infra.storage import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physicalrsi operator", description="Inspect and explicitly recover a local device")
    commands = parser.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="Save the device's identity, generation and current stop state")
    inspect.add_argument("--socket", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Write a reviewable recovery request; no device operation")
    prepare.add_argument("--inspection", type=Path, required=True)
    prepare.add_argument("--request-id", required=True)
    prepare.add_argument("--reason", required=True)
    prepare.add_argument("--evidence", type=Path, required=True, help="JSON object with operator inspection evidence")
    prepare.add_argument("--output", type=Path, required=True)
    recover = commands.add_parser("recover", help="Submit exactly the recovery request saved in this file")
    recover.add_argument("--socket", type=Path, required=True)
    recover.add_argument("--request", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("status", help="Read an existing request after an interrupted response")
    status.add_argument("--socket", type=Path, required=True)
    status.add_argument("--request-id", required=True)
    status.add_argument("--output", type=Path)
    for command in (inspect, recover, status):
        command.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise ValueError("Output already exists; choose a fresh artifact path before contacting the device")
        if args.action == "prepare":
            result = recovery_request(read_json(args.inspection), request_id=args.request_id,
                                      reason=args.reason, evidence=read_json(args.evidence))
        else:
            inputs = (read_json(args.request) if args.action == "recover" else
                      dict(request_id=args.request_id) if args.action == "status" else {})
            result = OperatorClient(args.socket).call(args.action, inputs, timeout_seconds=args.timeout)
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__, error=str(error),
            automatic_retry=False, next="Inspect the original request with 'physicalrsi operator status' after a timeout")), file=sys.stderr)
        return 1
