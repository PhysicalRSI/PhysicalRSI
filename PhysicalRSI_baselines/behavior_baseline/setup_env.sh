#!/usr/bin/env bash
# Install upstream BEHAVIOR in an isolated CPFS (or other large disk) runtime.
set -euo pipefail
runtime_root="${1:?Usage: setup_env.sh /absolute/runtime/root}"
runtime_root="$(realpath -m "$runtime_root")"
source_root="$runtime_root/sources/BEHAVIOR-1K"
expected_revision=a8247a8cc1633fe1ca0cc66aa07243d46c64f155
mkdir -p "$runtime_root"/{tmp,cache,logs,data}
export TMPDIR="$runtime_root/tmp"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export UV_CACHE_DIR="$runtime_root/cache/uv"
export HF_HOME="$runtime_root/cache/huggingface"
export XDG_CACHE_HOME="$runtime_root/cache"
export OMNIGIBSON_DATA_PATH="$runtime_root/data"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
if [[ ! -x "$runtime_root/env/bin/python" ]]; then
    echo "Create a Python 3.11 venv at $runtime_root/env first." >&2
    exit 1
fi
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$expected_revision" ]]; then
    echo "Expected BEHAVIOR v3.9.3-post2 ($expected_revision)." >&2
    exit 1
fi
if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
    echo "Upstream tracked sources must be unchanged." >&2
    exit 1
fi
source "$runtime_root/env/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
python -m pip install 'setuptools>=71,<81' wheel
cd "$source_root"
# The user requested environment installation. These are upstream's automated
# installation flags; assets and their key remain outside the code repository.
bash setup.sh --omnigibson --bddl --joylo --eval --dataset \
    --confirm-no-conda --accept-nvidia-eula --accept-dataset-tos
python -m pip freeze > "$runtime_root/environment.freeze.txt"
