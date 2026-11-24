#!/usr/bin/env bash
# D-028: where does the saturated engine thread's time go with the kernel and CUDA graphs? Four seeds
# at saturation (batch 64, 32 req/s offered), then bench.phase_report. No comparison arm: the
# reference throughput is the D-020 graph arm. Self-test before each seed, monitor throughout.
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
# Everything stays under ~/pagedserve_work. NOT YET RUN (the machine was off limits when written).
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d028
for f in results/machine-monitor-$TAG.jsonl results/phase-report-rtx5090.md; do
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
for seed in 0 1 2 3; do
  selftest sat-s$seed
  "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL \
      --out "results/d028-sat-rtx5090-seed$seed" --server-env results/env-engine.json \
      --batch-sizes 64 --rates 32 --kv-cache-gb 8 --seed "$seed" \
      --server-arg=--paged-attention-kernel --server-arg=--cuda-graphs --resume
  log "engine saturation (kernel + graphs) seed $seed exit $?"
done
"$ENGINE_PY" -m bench.phase_report --engine "results/d028-sat-rtx5090-seed{seed}" --seeds 0,1,2,3 \
    --out results/phase-report-rtx5090.md
log "phase report exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D028-DONE"
