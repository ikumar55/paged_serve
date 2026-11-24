#!/usr/bin/env bash
# D-025 diagnostic speed measurement (NOT the D-025 claim; see the plan in D-025): the real-mix
# decode step with and without the Triton kernel, then the engine at saturation (batch 64, 32 req/s
# offered) with and without it, four seeds, paired within each seed, alternating which runs first.
# Self-test before each pair, machine monitor throughout. Expects ~/pagedserve_work/tmp/d025-code.tgz
# and tmp/COMMIT as in gpu_d025_kernel.sh. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d025-speed
for f in results/profile-mix-d025-rtx5090.json results/machine-monitor-$TAG.jsonl; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done
selftest() {
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$TAG-$1.json" 2>&1
  rc=$?; log "self-test before $1 exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
log "code installed, commit $(cat COMMIT)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6

selftest mixprofile
"$ENGINE_PY" -m bench.profile_decode --model-dir $MODEL --batch-sizes 64 --steps 40 \
    --mix-seeds 0,1,2,3 --mix-batch 60 --mix-attention both --skip-profile \
    --out results/profile-mix-d025-rtx5090.json
log "mix profile exit $?"

for seed in 0 1 2 3; do
  if [ $((seed % 2)) -eq 0 ]; then order="gather kernel"; else order="kernel gather"; fi
  selftest sat-s$seed
  for variant in $order; do
    if [ "$variant" = kernel ]; then extra=(--server-arg=--paged-attention-kernel); else extra=(); fi
    "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL \
        --out "results/d025-sat-$variant-rtx5090-seed$seed" --server-env results/env-engine.json \
        --batch-sizes 64 --rates 32 --kv-cache-gb 8 --seed "$seed" "${extra[@]}" --resume
    log "engine saturation $variant seed $seed exit $?"
  done
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D025-SPEED-DONE"
