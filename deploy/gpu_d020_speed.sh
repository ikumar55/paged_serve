#!/usr/bin/env bash
# D-020 diagnostic speed measurement (NOT the D-020 claim; see the plan in D-020): the real-mix
# decode step with the Triton kernel without and with CUDA graphs, then the engine at saturation
# (batch 64, 32 req/s offered) with the kernel without and with graphs, four seeds, paired within
# each seed, alternating which runs first. Self-test before each pair, monitor throughout.
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d020-speed
for f in results/profile-mix-d020-rtx5090.json results/machine-monitor-$TAG.jsonl; do
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
    --mix-seeds 0,1,2,3 --mix-batch 60 --mix-attention triton-graphs --skip-profile \
    --out results/profile-mix-d020-rtx5090.json
log "mix profile exit $?"

for seed in 0 1 2 3; do
  if [ $((seed % 2)) -eq 0 ]; then order="eager graphs"; else order="graphs eager"; fi
  selftest sat-s$seed
  for variant in $order; do
    extra=(--server-arg=--paged-attention-kernel)
    [ "$variant" = graphs ] && extra=(--server-arg=--paged-attention-kernel --server-arg=--cuda-graphs)
    "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL \
        --out "results/d020-sat-$variant-rtx5090-seed$seed" --server-env results/env-engine.json \
        --batch-sizes 64 --rates 32 --kv-cache-gb 8 --seed "$seed" "${extra[@]}" --resume
    log "engine saturation kernel+$variant seed $seed exit $?"
  done
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D020-SPEED-DONE"
