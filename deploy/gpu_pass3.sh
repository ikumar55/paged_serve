#!/usr/bin/env bash
# Pass 3: a shared-prefix workload with the prefix cache on and off, for the engine and for vLLM.
# Workload (a modelling choice, not "representative"): 8 shared prefixes of 1024 tokens with unequal
# popularity (skew 1), plus a unique suffix of median 64 tokens. Seeds 0 to 2, the four
# configurations interleaved within each seed, a determinism self-test before each system.
# Everything stays under ~/pagedserve_work. A failed stage is logged and the pass carries on.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
RATES=4,8,16
PREFIX_ARGS=(--loadgen-arg=--shared-prefix-groups=8 --loadgen-arg=--shared-prefix-len=1024
             --loadgen-arg=--shared-prefix-skew=1 --loadgen-arg=--prompt-median=64)
log() { echo "== $(date -u +%FT%TZ) $*"; }
gpu_state() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }
selftest() {
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$1.json" 2>&1
  rc=$?; log "self-test before $1 exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}
engine() { # seed on|off
  extra=(); [ "$2" = off ] && extra=(--server-arg=--no-prefix-caching)
  "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL --out "results/prefix-engine-$2-rtx5090-seed$1" \
      --server-env results/env-engine.json --batch-sizes 32 --rates $RATES --kv-cache-gb 8 --seed "$1" \
      "${PREFIX_ARGS[@]}" "${extra[@]}" --resume
  log "engine cache $2 seed $1 exit $?"; }
vllm() { # seed on|off
  extra=(); [ "$2" = off ] && extra=(--server-arg=--no-enable-prefix-caching)
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL \
      --out "results/prefix-vllm-$2-rtx5090-seed$1" --server-env results/env-vllm.json --batch-sizes 32 \
      --rates $RATES --seed "$1" "${PREFIX_ARGS[@]}" "${extra[@]}" --resume
  log "vllm cache $2 seed $1 exit $?"; }

log "pass3 start; gpu: $(gpu_state)"
for seed in 0 1 2; do
  selftest prefix-engine-s$seed; engine $seed on; engine $seed off
  selftest prefix-vllm-s$seed;   vllm   $seed on; vllm   $seed off
  log "seed $seed done; gpu: $(gpu_state); free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
done
log "disk: $(du -sh "$PAGEDSERVE_WORK" | cut -f1) used by the work dir; $(df -h "$HOME" | tail -1 | awk '{print $4}') free"
log "PASS3-DONE"
