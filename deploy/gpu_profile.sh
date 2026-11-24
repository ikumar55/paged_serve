#!/usr/bin/env bash
# Profile one decode step of the engine at several batch sizes (bench.profile_decode), with the
# determinism self-test first and the machine monitor running throughout.
# Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "profile start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-profile.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-profile.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-profile.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
"$ENGINE_PY" -m bench.profile_decode --model-dir models/Llama-3.2-1B-Instruct \
    --batch-sizes 1,8,32,64 --prompt-len 256 --steps 40 --out results/profile-decode-rtx5090.json
log "profile exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "PROFILE-DONE"
