#!/usr/bin/env bash
# D-021 follow-up: the scheduler offline (no server) on the standard workload's own requests
# (lognormal lengths, seeds 0 to 3, all waiting at once), batch 64, prefix cache off then on within
# each seed (run 2: the first attempt ran out of GPU memory after four runs, a harness bug). Self-test first, machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "offline-workload start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-offline-workload-run2.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-offline-workload-run2.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-offline-workload-run2.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
"$ENGINE_PY" -m bench.profile_decode --model-dir models/Llama-3.2-1B-Instruct --skip-profile \
    --offline-workload-seeds 0,1,2,3 --offline-prefix-cache both --batch-sizes 64 \
    --out results/offline-workload-rtx5090.json
log "offline workload exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "OFFLINE-WORKLOAD-DONE"
