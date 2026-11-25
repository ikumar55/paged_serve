#!/usr/bin/env bash
# D-031: the decode-attention kernel in isolation (bench.kernel_microbench) and its lever test.
# One session of about 40 minutes: self-test, the GPU correctness tests of the unsplit and the split-KV
# kernels, the real-mix decode step with graphs (the lever test's denominator), the microbenchmark in
# the engine environment, then FlashAttention and FlashInfer in other environments (vLLM's bundled
# flash_attn; FlashInfer installed into a throwaway directory, 15 minutes at most), and the merged
# report. Whatever does not install or fails its gate is reported as such, never substituted.
# Not run yet (written 2026-11-25 without a GPU). Everything stays under ~/pagedserve_work.
#
# Expects ~/pagedserve_work/tmp/d025-code.tgz (a `git archive` of the commit to measure, with
# engine/triton_split_attention.py and bench/kernel_microbench.py) and tmp/COMMIT.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve" || exit 1
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
EXT_DIR="$PAGEDSERVE_WORK/d031-ext"
MODEL=models/Llama-3.2-1B-Instruct
TAG=d031
log() { echo "== $(date -u +%FT%TZ) $*"; }
for f in results/kernel-microbench-rtx5090.json results/kernel-microbench-rtx5090.md \
         results/kernel-microbench-vllm-env-rtx5090.json results/kernel-microbench-flashinfer-rtx5090.json \
         results/d031-step-rtx5090.json results/machine-monitor-$TAG.jsonl; do
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

# 1. the correctness tests of both Triton kernels (the split-KV one has never been compiled)
"$ENGINE_PY" -m pytest -m gpu tests/test_paged_attention_gpu.py tests/test_split_attention_gpu.py \
    -q -s -p no:cacheprovider > results/d031-gpu-tests.log 2>&1
log "gpu tests exit $? (see results/d031-gpu-tests.log)"

# 2. the real-mix decode step with graphs: the denominator of the lever test
"$ENGINE_PY" -m bench.profile_decode --model-dir "$MODEL" --batch-sizes 64 --steps 40 \
    --mix-seeds 0,1,2,3 --mix-batch 60 --mix-attention triton-graphs --skip-profile \
    --out results/d031-step-rtx5090.json
log "step profile exit $?"

# 3. the microbenchmark, engine environment
"$ENGINE_PY" -m bench.kernel_microbench --device cuda --out results/kernel-microbench-rtx5090.json \
    --step-profile results/d031-step-rtx5090.json
log "microbenchmark exit $?"

# 4. FlashAttention as bundled with vLLM (its own environment and PyTorch)
"$VLLM_PY" -m bench.kernel_microbench --device cuda --impls flash-attn-paged \
    --out results/kernel-microbench-vllm-env-rtx5090.json
log "vllm-environment microbenchmark exit $?"

# 5. FlashInfer, installed into a throwaway directory (the engine environment's PyTorch is kept);
#    15 minutes at most
EXTRA=()
if timeout 900 bash -c "'$ENGINE_PY' -m pip install --quiet --target '$EXT_DIR' --no-deps flashinfer-python       && '$ENGINE_PY' -m pip install --quiet --target '$EXT_DIR' einops ninja"       > "$PAGEDSERVE_WORK/d031-flashinfer-install.log" 2>&1; then
  PYTHONPATH="$EXT_DIR" "$ENGINE_PY" -m bench.kernel_microbench --device cuda --impls flashinfer       --out results/kernel-microbench-flashinfer-rtx5090.json
  log "flashinfer microbenchmark exit $?"
  EXTRA+=(results/kernel-microbench-flashinfer-rtx5090.json)
else
  log "flashinfer did not install within 15 minutes (see d031-flashinfer-install.log); reported as not available"
fi

# 6. the merged report (the main run's own report was written next to its JSON)
MERGE=()
[ -e results/kernel-microbench-vllm-env-rtx5090.json ] && MERGE+=(results/kernel-microbench-vllm-env-rtx5090.json)
MERGE+=("${EXTRA[@]}")
"$ENGINE_PY" -m bench.kernel_microbench --report results/kernel-microbench-rtx5090.json \
    --merge "${MERGE[@]}" --step-profile results/d031-step-rtx5090.json \
    --out results/kernel-microbench-merged-rtx5090.md
log "merged report exit $?"

"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-after-$TAG.json 2>&1
log "self-test after exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
rm -rf "$EXT_DIR"
log "D031-DONE"
