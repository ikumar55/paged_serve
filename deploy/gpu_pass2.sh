#!/usr/bin/env bash
# Pass 2: seeds 1-3 for all three systems (interleaved per seed), then seed 0 at the extra rates.
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
engine() { # seed rates dir
  "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL --out "$3" --server-env results/env-engine.json \
      --batch-sizes 32,64 --rates "$2" --kv-cache-gb 8 --seed "$1" --resume; log "engine $3 exit $?"; }
hf() {
  "$ENGINE_PY" -m bench.sweep --system hf-static --model $MODEL --out "$3" --server-env results/env-engine.json \
      --batch-sizes 32,64 --windows-ms 20 --rates "$2" --seed "$1" --resume; log "hf-static $3 exit $?"; }
vllm() {
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL --out "$3" \
      --server-env results/env-vllm.json --batch-sizes 32 --rates "$2" --seed "$1" --resume; log "vllm $3 exit $?"; }

log "pass2 start; gpu: $(gpu_state)"
for seed in 1 2 3; do
  selftest engine-s$seed;  engine $seed 4,8,12,16      results/engine-rtx5090-seed$seed
  selftest hf-s$seed;      hf     $seed 4,8,16         results/hf-static-rtx5090-seed$seed
  selftest vllm-s$seed;    vllm   $seed 4,8,16,24,32,48 results/vllm-rtx5090-seed$seed
  log "seed $seed done; gpu: $(gpu_state); free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
done
log "extra rates at seed 0"
selftest engine-s0x; engine 0 12          results/engine-rtx5090-seed0-extra
selftest vllm-s0x;   vllm   0 24,32,48    results/vllm-rtx5090-seed0-extra
log "disk: $(du -sh "$PAGEDSERVE_WORK" | cut -f1) used by the work dir; $(df -h "$HOME" | tail -1 | awk '{print $4}') free"
log "PASS2-DONE"
