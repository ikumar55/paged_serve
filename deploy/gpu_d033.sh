#!/usr/bin/env bash
# D-033: KL divergence and perplexity of the bf16 paths on real text, against Hugging Face fp32
# (bench.kl_perplexity). 200 windows of 256 tokens, 127 teacher-forced decode steps each; the engine's
# fp32 and bf16 gather paths, the Triton kernel and the CUDA-graph path against Hugging Face fp32, with
# Hugging Face bf16 as the yardstick. Self-test first, machine monitor throughout.
#
# Expects ~/pagedserve_work/tmp/d025-code.tgz and tmp/COMMIT as in gpu_d025_kernel.sh, BUT the archive
# must also contain docs/ and README.md (the fixed text comes from the repository's own documents):
#   git archive --format=tar.gz -o d025-code.tgz HEAD engine bench tests router deploy docs README.md pyproject.toml
# Everything stays under ~/pagedserve_work. NOT YET RUN (the machine was not available when written).
#
# WIDE=1 bash gpu_d033.sh runs only the 400-window widening (extended text set), for use if the first
# run is inconclusive (D-033, amendments 1 and 2).
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
log() { echo "== $(date -u +%FT%TZ) $*"; }
WIDE="${WIDE:-0}"
if [ "$WIDE" = 1 ]; then TAG=d033-wide; LABEL=rtx5090-wide; ARGS=(--windows 400 --text-set extended)
else TAG=d033; LABEL=rtx5090; ARGS=(); fi
for f in results/selftest-before-$TAG.json results/machine-monitor-$TAG.jsonl \
         results/kl-perplexity-$LABEL.json; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
[ -f docs/decisions.md ] && [ -f README.md ] || { log "STAGE-FAILED the archive lacks docs/ or README.md"; exit 1; }
log "code installed, commit $(cat COMMIT)"
nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-$TAG.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }

"$ENGINE_PY" -m pytest tests/test_kl_perplexity.py -q -p no:cacheprovider 2>&1 | tail -2
"$ENGINE_PY" -m bench.kl_perplexity --model-dir $MODEL --out results --label $LABEL "${ARGS[@]}"
log "kl-perplexity exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D033-DONE"
