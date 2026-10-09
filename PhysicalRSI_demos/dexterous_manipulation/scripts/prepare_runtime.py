"""Prepare the pinned GPT-as-Policy runtime in a new, external directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--source", help="Optional local upstream clone for offline preparation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / "provenance.json").read_text())["gpt_as_policy"]
    destination = Path(args.destination).expanduser().resolve()
    if destination.exists():
        parser.error("Destination must be new; existing source is never patched in place")
    patch = root / pin["patch"]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(patch) != pin["patch_sha256"]:
        raise ValueError("Runtime patch differs from provenance")
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--no-hardlinks",
                    args.source or pin["repository"], str(destination)], check=True)
    def git(*command, **kwargs):
        return subprocess.run(["git", "-C", str(destination), *command], check=True, **kwargs)
    git("sparse-checkout", "init", "--no-cone")
    git("sparse-checkout", "set", "--no-cone", "--stdin", text=True,
        input="/hybrid_rollout/__init__.py\n/hybrid_rollout/robodojo/\n/LICENSE\n/THIRD_PARTY_NOTICES.md\n/licenses/\n")
    git("checkout", "--detach", pin["commit"])
    git("apply", "--check", str(patch))
    git("apply", str(patch))
    for name, expected in pin["expected_runtime_sha256"].items():
        if sha(destination / name) != expected:
            raise ValueError("Prepared runtime differs from used source: " + name)
    print(json.dumps(dict(runtime_root=str(destination), commit=pin["commit"],
                         verified_files=len(pin["expected_runtime_sha256"])), indent=2))


if __name__ == "__main__":
    main()
