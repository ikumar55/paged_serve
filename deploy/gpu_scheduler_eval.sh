#!/usr/bin/env bash
# Check that the scheduler (batching, paging, prefix cache, preemption) serves the same text as
# Hugging Face on this GPU: bench.scheduler_eval in bf16 (what the benchmarks used) and fp32.
# Waits for the vLLM ceiling sweep to finish first (never two GPU jobs at once). Self-test first,
# machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "scheduler-eval waiting for the ceiling sweep"
while pgrep -u "$(whoami)" -f '^bash ceiling.sh' > /dev/null; do sleep 15; done
log "scheduler-eval start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-scheduler-eval.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-scheduler-eval.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-scheduler-eval.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
for dtype in bfloat16 float32; do
  label=rtx5090-${dtype/bfloat16/bf16}
  label=${label/float32/fp32}
  "$ENGINE_PY" -m bench.scheduler_eval --model-dir models/Llama-3.2-1B-Instruct --out results \
      --label "$label" --device cuda --dtype "$dtype" --prompts 50 --tokens 64
  log "scheduler-eval $dtype exit $?"
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "SCHEDEVAL-DONE"
