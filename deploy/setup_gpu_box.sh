#!/usr/bin/env bash
# Prepare a fresh rented Linux GPU machine for pagedserve measurements.
#
# Run successfully once, on an Oxmiq Capsule RTX 5090 machine on 2026-11-23 (with
# ALLOW_OLDER_DRIVER=1, because that fleet's drivers advertise CUDA 13.0); never on a rented
# instance. Read it before running it; it is short.
#
# What it does, in order, stopping at the first failure:
#   1. records the GPU, driver and OS and refuses to continue if there is no usable NVIDIA GPU;
#   2. checks the driver supports the CUDA runtime that the pinned torch build needs;
#   3. creates the engine environment from constraints.txt (the CUDA build of torch);
#   4. optionally creates a separate vLLM environment (vLLM pins a different torch);
#   5. runs the D-016 determinism self-test and refuses to go on if the GPU fails it.
#
# It does NOT download model weights (that needs your Hugging Face token and licence acceptance;
# see the README) and it never writes a token anywhere.
#
# Usage:
#   bash deploy/setup_gpu_box.sh                 # engine environment only
#   VLLM_VERSION=0.34.0 bash deploy/setup_gpu_box.sh   # also build the vLLM environment
#
# Environment variables:
#   TORCH_CUDA_INDEX  PyTorch wheel index for the CUDA build (default: cu132, matching the pins)
#   VENV_DIR          where the environments go (default: $HOME/venvs)
#   VLLM_VERSION      if set, install this vLLM version into its own environment
#   ALLOW_OLDER_DRIVER  set to 1 to continue when the driver advertises an older CUDA than the
#                     pinned torch build (the self-test still has to pass)
set -euo pipefail

TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu132}"
VENV_DIR="${VENV_DIR:-$HOME/venvs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.12}"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

say "1. Hardware"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[ "$GPU_COUNT" -ge 1 ] || die "no GPU visible"
echo "GPUs visible: $GPU_COUNT (the comparison should use exactly one; set CUDA_VISIBLE_DEVICES=0)"
df -h "$HOME" | tail -1
free -g | sed -n 2p

say "2. Driver vs CUDA runtime"
# nvidia-smi prints the highest CUDA version the driver supports in its header.
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
echo "Driver supports up to CUDA $DRIVER_CUDA"
WANT="${TORCH_CUDA_INDEX#cu}"           # e.g. 132 -> 13.2
WANT_MAJOR="${WANT:0:2}"
WANT_MINOR="${WANT:2}"
WANT_VERSION="$WANT_MAJOR.$WANT_MINOR"
# D-016: a runtime newer than the driver advertises can run and still misbehave, so refuse it,
# unless ALLOW_OLDER_DRIVER=1 is set. The override exists because D-017 ran this same cu132 stack
# on a driver that advertises CUDA 13.0 and it passed the self-test and the greedy evaluation; with
# the override, step 5 (the self-test) is the only gate, so do not skip it and say so in D-018.
if [ "$(printf '%s\n%s\n' "$WANT_VERSION" "$DRIVER_CUDA" | sort -V | head -1)" != "$WANT_VERSION" ]; then
    if [ "${ALLOW_OLDER_DRIVER:-0}" = "1" ]; then
        echo "WARNING: torch wants CUDA $WANT_VERSION, driver advertises $DRIVER_CUDA; continuing because" \
             "ALLOW_OLDER_DRIVER=1. The self-test below decides whether this machine is usable."
    else
        die "torch wants CUDA $WANT_VERSION but the driver only supports $DRIVER_CUDA. Pick another \
TORCH_CUDA_INDEX (and record it) or another machine, or set ALLOW_OLDER_DRIVER=1 (see the comment above)."
    fi
fi

say "3. Engine environment"
command -v "$PYTHON" >/dev/null || die "$PYTHON not found (Python 3.12 is the project's version)"
mkdir -p "$VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR/pagedserve"
ENGINE_PY="$VENV_DIR/pagedserve/bin/python"
"$ENGINE_PY" -m pip install --upgrade pip
"$ENGINE_PY" -m pip install -c "$REPO_DIR/constraints.txt" torch \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX"
"$ENGINE_PY" -m pip install -c "$REPO_DIR/constraints.txt" -e "$REPO_DIR[dev]"
"$ENGINE_PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
PY

if [ -n "${VLLM_VERSION:-}" ]; then
    say "4. vLLM environment ($VLLM_VERSION, separate: it pins its own torch)"
    "$PYTHON" -m venv "$VENV_DIR/vllm"
    "$VENV_DIR/vllm/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/vllm/bin/python" -m pip install "vllm==$VLLM_VERSION"
    "$VENV_DIR/vllm/bin/python" -m pip freeze | grep -iE '^(vllm|torch)==' || true
else
    say "4. vLLM environment skipped (set VLLM_VERSION to build it)"
fi

say "5. Determinism self-test (D-016)"
cd "$REPO_DIR"
"$ENGINE_PY" -m bench.gpu_selftest || die "the GPU FAILED the self-test: do not measure on it; release the instance"

say "Ready"
echo "Activate with: source $VENV_DIR/pagedserve/bin/activate"
echo "Next: record the environment with   python -m bench.env_capture --out results"
