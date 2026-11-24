#!/usr/bin/env bash
# The scheduler evaluation on the exact 50 prompts of the recorded plain-engine greedy evaluation
# (results/greedy-eval-rtx5090-bf16.json), so the two can be compared prompt for prompt. bf16 twice
# (is the scheduler's output reproducible from run to run?) and fp32 once. Self-test first, machine
# monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
FROZEN=results/greedy-eval-rtx5090-bf16.json
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "frozen start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-frozen.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-frozen.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-frozen.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
for run in "bf16-frozen bfloat16" "bf16-frozen-repeat bfloat16" "fp32-frozen float32"; do
  set -- $run
  "$ENGINE_PY" -m bench.scheduler_eval --model-dir models/Llama-3.2-1B-Instruct --out results \
      --label "rtx5090-$1" --device cuda --dtype "$2" --tokens 64 --prompts-from "$FROZEN"
  log "scheduler-eval $1 exit $?"
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "FROZEN-DONE"
