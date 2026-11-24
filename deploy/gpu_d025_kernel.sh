#!/usr/bin/env bash
# D-025, GPU session 1: put the committed code on the box, then the kernel's correctness gate
# (criterion (a)(i)) on the real GPU. Self-test first, machine monitor throughout.
# Expects ~/pagedserve_work/tmp/d025-code.tgz (a `git archive` of the committed code: engine bench
# tests router deploy pyproject.toml) and ~/pagedserve_work/tmp/COMMIT (its commit id).
# Everything stays under ~/pagedserve_work. Result files are write-once.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve"
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
log() { echo "== $(date -u +%FT%TZ) $*"; }
TAG=d025-kernel-gate
for f in results/$TAG.log results/selftest-before-$TAG.json results/machine-monitor-$TAG.jsonl; do
  [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
done

log "$TAG start; gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
sha256sum "$PAGEDSERVE_WORK/tmp/d025-code.tgz"
tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
log "code installed, commit $(cat COMMIT); triton $("$ENGINE_PY" -c 'import triton; print(triton.__version__)')"

nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out results/machine-monitor-$TAG.jsonl \
    --interval 2 > "$PAGEDSERVE_WORK/monitor-$TAG.log" 2>&1 < /dev/null &
sleep 6
"$ENGINE_PY" -m bench.gpu_selftest > results/selftest-before-$TAG.json 2>&1
rc=$?; log "self-test exit $rc"
[ $rc -eq 0 ] || { log "STAGE-FAILED self-test: not measuring on this GPU"; exit 1; }

"$ENGINE_PY" -m pytest tests/test_paged_attention.py -q -p no:cacheprovider > results/$TAG.log 2>&1
log "CPU tests of the new code exit $?"
"$ENGINE_PY" -m pytest -m gpu tests/test_paged_attention_gpu.py -s -v -p no:cacheprovider \
    >> results/$TAG.log 2>&1
log "kernel gate tests exit $?"
sleep 6
pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
sleep 4
log "D025-KERNEL-GATE-DONE"
