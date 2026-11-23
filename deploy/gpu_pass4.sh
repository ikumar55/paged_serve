#!/usr/bin/env bash
# Pass 4: a repeat of part of pass 2 (same seeds, same workloads, same configurations) while
# bench.machine_monitor is running, to show how reproducible the earlier, unmonitored results are.
# Waits for pass 3 to finish first (never two measurements at once).
# Everything stays under ~/pagedserve_work. A failed stage is logged and the pass carries on.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
gpu_state() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }
selftest() {
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$1.json" 2>&1
  rc=$?; log "self-test before $1 exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}

log "pass4 waiting for pass3 to finish"
while pgrep -u "$(whoami)" -f '^bash pass3.sh' > /dev/null; do sleep 20; done
log "pass4 start; gpu: $(gpu_state)"
for seed in 1 2; do
  selftest repro-engine-s$seed
  "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL --out "results/repro-engine-rtx5090-seed$seed" \
      --server-env results/env-engine.json --batch-sizes 32,64 --rates 8,16 --kv-cache-gb 8 \
      --seed "$seed" --resume; log "engine repro seed $seed exit $?"
  selftest repro-hf-s$seed
  "$ENGINE_PY" -m bench.sweep --system hf-static --model $MODEL --out "results/repro-hf-static-rtx5090-seed$seed" \
      --server-env results/env-engine.json --batch-sizes 32,64 --windows-ms 20 --rates 8,16 \
      --seed "$seed" --resume; log "hf-static repro seed $seed exit $?"
  selftest repro-vllm-s$seed
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL \
      --out "results/repro-vllm-rtx5090-seed$seed" --server-env results/env-vllm.json --batch-sizes 32 \
      --rates 16,48 --seed "$seed" --resume; log "vllm repro seed $seed exit $?"
  log "seed $seed done; gpu: $(gpu_state); free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
done
log "PASS4-DONE"
