#!/usr/bin/env bash
# D-020 prerequisite: profile one decode step of a batch of 60 sequences whose context lengths are
# drawn from the standard workload (four seeds), and, in the same session, the fixed-length batch 64
# of D-003 for comparison. Self-test first, machine monitor throughout.
# Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }

log "mix-profile start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-mix-profile.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-mix-profile.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-mix-profile.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }
"$ENGINE_PY" -m bench.profile_decode --model-dir models/Llama-3.2-1B-Instruct \
    --batch-sizes 64 --prompt-len 256 --steps 40 --mix-seeds 0,1,2,3 --mix-batch 60 \
    --out results/profile-mix-rtx5090.json
log "mix profile exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "MIX-PROFILE-DONE"
