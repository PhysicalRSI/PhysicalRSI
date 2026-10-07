"""Retain pinned ASPIRE math and validation; retry only known solver rejection.

The fallback changes the numerical initial guess, not simulator state. Every
returned solution still passes the original finite, joint-limit and FK checks.
"""
import json
from pathlib import Path
import sys

from PhysicalRSI_core.infra.storage import atomic_json
from . import aspire_ik_worker as worker

NOMINAL_HOME = (0., -.785398, 0., -2.356194, 0., 1.570796, .785398)
_single_solve = worker._worker


def solve_with_fallback(request, *, solve=None):
    solve = _single_solve if solve is None else solve
    attempts = []
    for name, joints in (("current", request["joints"]), ("nominal-home", list(NOMINAL_HOME))):
        try:
            result = solve({**request, "joints": list(joints)})
        except (RuntimeError, ValueError) as exc:
            known = ((type(exc) is RuntimeError and str(exc) == "IK target did not converge")
                     or (type(exc) is ValueError and str(exc) == "Invalid IK solution"))
            if not known:
                raise
            attempts.append({"seed": name, "rejected": str(exc)})
            continue
        return {**result, "seed_attempts": attempts + [{"seed": name, "accepted": True}]}
    print(json.dumps({"id": request.get("id"), "seed_attempts": attempts}), file=sys.stderr, flush=True)
    raise RuntimeError("IK target did not converge")


def main():
    directory = Path(sys.argv[1])
    if not directory.is_dir():
        raise ValueError("Existing owned worker directory required")
    index = 0

    def recorded(request):
        nonlocal index
        # The transport validates and removes its sequence ID before solving.
        path = directory / f"request-{index:04d}.json"
        if path.exists():
            raise ValueError("Refusing to overwrite IK request evidence")
        atomic_json(path, {"id": index, **request})
        try:
            result = solve_with_fallback(request)
            atomic_json(directory / f"solution-{index:04d}.json", result)
            return result
        finally:
            index += 1

    worker._worker = recorded
    worker.main()


if __name__ == "__main__":
    main()
