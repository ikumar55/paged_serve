#!/usr/bin/env bash
# Pass 5 (from the independent review): capacity, not arrival-limited rates. At 16 req/s offered the
# delivered throughput is capped by the arrivals (D-019), so this offers 32 req/s to the engine and
# Hugging Face and 64 req/s to vLLM (above what each can serve), four seeds, the three systems
# interleaved within each seed so every ratio is paired. Plus: vLLM in eager mode (does CUDA graph
# capture explain its advantage?), the scheduler evaluated on shared-prefix prompts so the prefix
# cache is actually hit (fp32 and bf16), the fp32 probe, and the provenance records (weight
# hashes, dependency freezes, hashes of the exact code that ran).
# Waits for any earlier GPU job to finish (never two at once); self-test before every system;
# machine monitor throughout. Everything stays under ~/pagedserve_work.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
gpu_state() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; }
selftest() {
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$1.json" 2>&1
  rc=$?; log "self-test before $1 exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}

log "pass5 waiting for earlier GPU jobs"
while pgrep -u "$(whoami)" -f '^bash (frozen|followup|numerics|sched|ceiling).sh' > /dev/null; do sleep 15; done
log "pass5 start; gpu: $(gpu_state)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-pass5.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-pass5.log" 2>&1 < /dev/null &
sleep 6

# provenance: what exactly ran here
( cd "$PAGEDSERVE_WORK/paged_serve" && sha256sum engine/*.py bench/*.py bench/baselines/*.py deploy/*.sh \
    router/*.py 2>/dev/null ) > results/box-code-hashes-pass5.txt
( cd "$MODEL" && sha256sum * ) > results/weights-sha256.txt
"$ENGINE_PY" -m pip freeze > results/env-freeze-engine.txt 2>&1
"$VLLM_PY" -m pip freeze > results/env-freeze-vllm.txt 2>&1
log "provenance recorded"

selftest pass5-probe
"$ENGINE_PY" -m bench.fp32_probe --device cuda > results/fp32-probe-rtx5090.json 2> results/fp32-probe-rtx5090.err
log "fp32 probe exit $?"
for spec in "float32 prefix-fp32" "bfloat16 prefix-bf16"; do
  set -- $spec
  "$ENGINE_PY" -m bench.scheduler_eval --model-dir "$MODEL" --out results --label "rtx5090-$2" \
      --device cuda --dtype "$1" --prompts 50 --tokens 64 --shared-prefix-groups 8 \
      --shared-prefix-len 1024 --shared-prefix-suffix 64
  log "scheduler-eval $2 exit $?"
done

for seed in 0 1 2 3; do
  selftest sat-engine-s$seed
  "$ENGINE_PY" -m bench.sweep --system engine --model $MODEL --out "results/sat-engine-rtx5090-seed$seed" \
      --server-env results/env-engine.json --batch-sizes 32,64 --rates 32 --kv-cache-gb 8 \
      --seed "$seed" --resume; log "engine sat seed $seed exit $?"
  selftest sat-hf-s$seed
  "$ENGINE_PY" -m bench.sweep --system hf-static --model $MODEL --out "results/sat-hf-static-rtx5090-seed$seed" \
      --server-env results/env-engine.json --batch-sizes 64 --windows-ms 20 --rates 32 \
      --seed "$seed" --resume; log "hf-static sat seed $seed exit $?"
  selftest sat-vllm-s$seed
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL \
      --out "results/sat-vllm-rtx5090-seed$seed" --server-env results/env-vllm.json --batch-sizes 32 \
      --rates 64 --seed "$seed" --resume; log "vllm sat seed $seed exit $?"
  log "seed $seed done; gpu: $(gpu_state); free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
done

for seed in 0 1; do
  selftest sat-vllm-eager-s$seed
  "$ENGINE_PY" -m bench.sweep --system vllm --vllm-python "$VLLM_PY" --model $MODEL \
      --out "results/sat-vllm-eager-rtx5090-seed$seed" --server-env results/env-vllm.json \
      --batch-sizes 32 --rates 16,64 --seed "$seed" --server-arg=--enforce-eager --resume
  log "vllm eager seed $seed exit $?"
done
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "disk: $(du -sh "$PAGEDSERVE_WORK" | cut -f1) used by the work dir; $(df -h "$HOME" | tail -1 | awk '{print $4}') free"
log "PASS5-DONE"
