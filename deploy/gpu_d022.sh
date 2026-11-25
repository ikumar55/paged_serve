#!/usr/bin/env bash
# D-022: the router policies on REAL engine replicas (bench.router_compare --backend llama).
# Three real engine.server replicas (bf16) on the ONE GPU behind the router, four policies, four seeds
# interleaved within each seed, a fresh fleet per run, 600 requests per run at 6 req/s, 32 shared
# prefixes of 1,024 tokens, 0.5 GB of KV cache per replica. The replicas, the router and the load
# generator share the GPU and the CPU: a confound stated in every result (D-022 and its amendment 1).
# 16 runs of about 3 to 4 minutes each: about one hour. Self-test before and after, monitor throughout.
#
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
# Everything stays under ~/pagedserve_work. NOT YET RUN (the machine was not available when written).
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d022
OUT=results/router-compare-real-rtx5090.json
for f in $OUT "${OUT%.json}.md" results/selftest-before-$TAG.json results/selftest-after-$TAG.json \
         results/machine-monitor-$TAG.jsonl; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
log "open-file limit for this session: $(ulimit -n)"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
log "code installed, commit $(cat COMMIT)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-$TAG.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }

"$ENGINE_PY" -m pytest tests/test_router_compare.py -q -p no:cacheprovider 2>&1 | tail -2
"$ENGINE_PY" -m bench.router_compare --out $OUT --label rtx5090-d022 \
    --backend llama --model-dir $MODEL --dtype bfloat16 --warmup-requests 2 \
    --replicas 3 --max-batch-size 32 --block-size 16 --kv-cache-gb 0.5 \
    --load-cap 1000 --poll-interval 0.5 --startup-timeout 300 \
    --rate 6 --num-requests 600 --shared-prefix-groups 32 --shared-prefix-len 1024 \
    --shared-prefix-skew 1 --prompt-median 64 --prompt-max 256 --output-median 32 \
    --output-max 128 --vocab-size 100000 --timeout 120 --seeds 0,1,2,3 \
    --policies round-robin,least-outstanding,approx-prefix,precise-prefix
log "router_compare exit $?"
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-after-$TAG.json 2>&1
log "self-test after exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D022-DONE"
