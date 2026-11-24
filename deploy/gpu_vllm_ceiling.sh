#!/usr/bin/env bash
# Find where vLLM stops keeping up: pass 2 reached 48 req/s without saturating. Seed 0, the standard
# workload, rates above that, max sequences 32 and 64. Self-test first, machine monitor throughout.
# Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "vllm ceiling start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-ceiling.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-ceiling.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-vllm-ceiling.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
"$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model models/Llama-3.2-1B-Instruct \
    --out results/vllm-rtx5090-ceiling --server-env results/env-vllm.json --batch-sizes 32,64 \
    --rates 64,96,128 --seed 0 --resume
log "vllm ceiling sweep exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "CEILING-DONE"
