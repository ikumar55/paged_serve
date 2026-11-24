#!/usr/bin/env bash
# fp32 numerics on this GPU: the GPU tests under the contract of D-019 (engine equals Hugging Face
# on the same GPU; no further from the CPU reference than Hugging Face's own GPU run), and the
# float64 truth report (each fp32 implementation's error against float64).
# Waits for the scheduler evaluation to finish first (never two GPU jobs at once).
# Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "numerics waiting for the scheduler evaluation"
while pgrep -u "$(whoami)" -f '^bash (ceiling|sched).sh' > /dev/null; do sleep 15; done
log "numerics start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
"$ENGINE_PY" -m pytest tests/test_real_llama_gpu.py -m gpu -s -q -p no:cacheprovider \
    > results/gpu-tests-rtx5090.txt 2>&1
log "gpu tests exit $?"
"$ENGINE_PY" -m tests.fp64_truth_report > results/fp64-truth-rtx5090.txt 2>&1
log "fp64 truth report exit $?"
log "NUMERICS-DONE"
