#!/usr/bin/env bash
# D-026: how far are the bf16 gather path and the bf16 Triton kernel from an fp32 reference, position
# by position (bench.bf16_logit_error)? Four prompt sets (the 50 frozen prompts; the shared-prefix
# prompts of D-025, seeds 0, 1, 2), 63 decode steps each, then the pooled report with D-026's
# criteria. Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
# Self-test first, machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
FROZEN=results/greedy-eval-rtx5090-bf16.json
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d026
for f in results/selftest-before-$TAG.json results/machine-monitor-$TAG.jsonl \
         results/bf16-logit-error-report-rtx5090.json; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
log "code installed, commit $(cat COMMIT)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-$TAG.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }

"$ENGINE_PY" -m pytest tests/test_bf16_logit_error.py -q -p no:cacheprovider 2>&1 | tail -2
"$ENGINE_PY" -m bench.bf16_logit_error --model-dir $MODEL --out results --label rtx5090-frozen \
    --prompts-from "$FROZEN" --kv-cache-gb 5
log "frozen exit $?"
for seed in 0 1 2; do
  "$ENGINE_PY" -m bench.bf16_logit_error --model-dir $MODEL --out results \
      --label rtx5090-prefix-s$seed --prompts 50 --shared-prefix-groups 8 --shared-prefix-len 1024 \
      --shared-prefix-suffix 64 --seed $seed --kv-cache-gb 5
  log "prefix seed $seed exit $?"
done
"$ENGINE_PY" -m bench.bf16_logit_error report results/bf16-logit-error-rtx5090-*.json \
    --out results/bf16-logit-error-report-rtx5090.json
log "report exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D026-DONE"
