#!/usr/bin/env bash
# D-025, GPU session 3: criterion (a) again after the kernel change (softmax weights rounded to
# the value dtype before the sum over V; see D-025). The kernel gate tests again (session 1 had a test bug,
# fixed since; its log is kept as it was), then the scheduler evaluations with the Triton kernel
# on: fp32 on the frozen prompts and on the shared-prefix prompts (cache hits), 50 of 50 identical
# to Hugging Face required in all three configurations; bf16 the same two ways, every divergence
# within one bf16 logit step (0.125). Same prompts and settings as the recorded gather-path runs
# (gpu_scheduler_frozen.sh, gpu_pass5.sh), so they can be compared prompt for prompt.
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
# Self-test first, machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
FROZEN=results/greedy-eval-rtx5090-bf16.json
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d025-eval-v2
for f in results/d025-kernel-gate-3.log results/selftest-before-$TAG.json \
         results/machine-monitor-$TAG.jsonl; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
sha256sum "$PAGEDSERVE_WORK/tmp/d025-code.tgz"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
log "code installed, commit $(cat COMMIT)"

nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-$TAG.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }

"$ENGINE_PY" -m pytest -m gpu tests/test_paged_attention_gpu.py -s -v -p no:cacheprovider \
    > results/d025-kernel-gate-3.log 2>&1
log "kernel gate tests exit $?"

# the bf16 runs are done twice, to see whether the near-tie divergences reproduce from run to run
for spec in "float32 fp32-frozen-d025kernelv2 frozen" "float32 fp32-prefix-d025kernelv2 prefix" \
            "bfloat16 bf16-frozen-d025kernelv2 frozen" "bfloat16 bf16-prefix-d025kernelv2 prefix" \
            "bfloat16 bf16-frozen-d025kernelv2-repeat frozen" \
            "bfloat16 bf16-prefix-d025kernelv2-repeat prefix"; do
  set -- $spec
  if [ "$3" = frozen ]; then
    extra=(--prompts-from "$FROZEN")
  else
    extra=(--prompts 50 --shared-prefix-groups 8 --shared-prefix-len 1024 --shared-prefix-suffix 64)
  fi
  "$ENGINE_PY" -m bench.scheduler_eval --model-dir "$MODEL" --out results \
      --label "rtx5090-$2" --device cuda --dtype "$1" --tokens 64 --paged-attention-kernel "${extra[@]}"
  log "scheduler-eval $2 exit $?"
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D025-EVAL-V2-DONE"
