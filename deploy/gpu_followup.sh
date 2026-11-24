#!/usr/bin/env bash
# Two follow-up measurements, self-test first and the machine monitor throughout:
#  1. bench.scheduler_eval in bf16 again, now recording for every divergence from Hugging Face how
#     far apart the reference's own logits put the two tokens (tie, or a wrong token?);
#  2. the scheduler run offline (no server) at max batch 32 and 64, to see where its time goes and
#     how its tokens per second compare with what the server delivers at saturation.
# Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "followup start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-followup.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-followup.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-followup.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
"$ENGINE_PY" -m bench.scheduler_eval --model-dir models/Llama-3.2-1B-Instruct --out results \
    --label rtx5090-bf16-gaps --device cuda --dtype bfloat16 --prompts 50 --tokens 64
log "scheduler-eval with gaps exit $?"
"$ENGINE_PY" -m bench.profile_decode --model-dir models/Llama-3.2-1B-Instruct --skip-profile \
    --offline-requests 128 --offline-new-tokens 128 --prompt-len 256 --batch-sizes 32,64 \
    --out results/offline-scheduler-rtx5090.json
log "offline scheduler exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "FOLLOWUP-DONE"
