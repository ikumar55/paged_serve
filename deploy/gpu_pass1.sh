#!/usr/bin/env bash
# Pass 1 (seed 0): engine, HF static baseline, then vLLM. Everything stays under ~/pagedserve_work.
# Continues past a failed stage so one problem does not waste the rest; every stage logs its exit code.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
RATES=2,4,8,16
log() { echo "== $(date -u +%FT%TZ) $*"; }
selftest() {
  log "self-test before $1"
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$1.json" 2>&1
  rc=$?; log "self-test exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}
gpu_state() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }

log "pass1 start; gpu: $(gpu_state)"
selftest engine
log "engine sweep (batch 32,64; rates $RATES)"
"$ENGINE_PY" -m bench.sweep --system engine --model $MODEL --out results/engine-rtx5090-p1 \
    --server-env results/env-engine.json --batch-sizes 32,64 --rates $RATES --kv-cache-gb 8 --seed 0 --resume
log "engine sweep exit $?"

selftest hf
log "hf-static sweep (batch 16,32,64; window 20 ms; rates $RATES)"
"$ENGINE_PY" -m bench.sweep --system hf-static --model $MODEL --out results/hf-static-rtx5090-p1 \
    --server-env results/env-engine.json --batch-sizes 16,32,64 --windows-ms 20 --rates $RATES --seed 0 --resume
log "hf-static sweep exit $?"

log "install vLLM 0.34.0 in its own environment"
python3.12 -m venv "$PAGEDSERVE_WORK/venvs/vllm" && "$VLLM_PY" -m pip install --upgrade pip \
    && "$VLLM_PY" -m pip install "vllm==0.34.0"
rc=$?; log "vllm install exit $rc"
if [ $rc -eq 0 ]; then
  "$VLLM_PY" -m pip freeze | grep -iE '^(vllm|torch)=='
  pip cache purge > /dev/null 2>&1
  "$VLLM_PY" -m bench.env_capture > results/env-vllm.json 2> results/env-vllm.err; log "env-vllm exit $?"
  selftest vllm
  log "vllm sweep (max sequences 32,64; rates $RATES)"
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL --out results/vllm-rtx5090-p1 \
      --server-env results/env-vllm.json --batch-sizes 32,64 --rates $RATES --seed 0 --resume
  log "vllm sweep exit $?"
fi
log "disk: $(du -sh "$PAGEDSERVE_WORK" | cut -f1) used by the work dir; $(df -h "$HOME" | tail -1 | awk '{print $4}') free"
log "PASS1-DONE"
