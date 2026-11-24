#!/usr/bin/env bash
# D-020 amendment 2, GPU session 1: correctness gates of the CUDA-graph decode step, before any
# speed is measured. The GPU tests (real capture; graph against eager bit for bit, padded steps),
# then bench.scheduler_eval with the Triton kernel AND --cuda-graphs on the same prompts as the
# recorded eager-kernel runs (D-025 session 2), so output hashes can be compared run for run.
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh.
# Self-test first, machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
FROZEN=results/greedy-eval-rtx5090-bf16.json
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d020-gates
for f in results/$TAG-gpu-tests.log results/selftest-before-$TAG.json \
         results/machine-monitor-$TAG.jsonl; do
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

"$ENGINE_PY" -m pytest tests/test_cuda_graph.py tests/test_paged_attention.py -q -p no:cacheprovider 2>&1 | tail -2
"$ENGINE_PY" -m pytest -m gpu tests/test_cuda_graph_gpu.py -s -v -p no:cacheprovider \
    > results/$TAG-gpu-tests.log 2>&1
rc=$?; log "graph GPU tests exit $rc"
tail -5 results/$TAG-gpu-tests.log | cut -c1-200
[ $rc -eq 0 ] || { log "STAGE-FAILED graph GPU tests: not running the evaluations"; \
  pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'; exit 1; }

for spec in "float32 fp32-frozen-d020graphs frozen" "float32 fp32-prefix-d020graphs prefix" \
            "bfloat16 bf16-frozen-d020graphs frozen" "bfloat16 bf16-prefix-d020graphs prefix"; do
  set -- $spec
  if [ "$3" = frozen ]; then
    extra=(--prompts-from "$FROZEN")
  else
    extra=(--prompts 50 --shared-prefix-groups 8 --shared-prefix-len 1024 --shared-prefix-suffix 64)
  fi
  "$ENGINE_PY" -m bench.scheduler_eval --model-dir "$MODEL" --out results \
      --label "rtx5090-$2" --device cuda --dtype "$1" --tokens 64 --paged-attention-kernel \
      --cuda-graphs "${extra[@]}"
  log "scheduler-eval $2 exit $?"
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D020-GATES-DONE"
