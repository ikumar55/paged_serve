#!/usr/bin/env bash
# Shared by deploy/gpu_d030_a.sh and deploy/gpu_d030_b.sh (D-030, the matched comparison with vLLM).
# Source it; do not run it. Everything stays under ~/pagedserve_work. NOT YET RUN.
#
# Expects ~/pagedserve_work/tmp/d025-code.tgz (a `git archive` of the commit to measure, with bench/
# including bench/null_server.py, bench/sharded.py and bench/matched_report.py) and tmp/COMMIT, as in
# gpu_d025_kernel.sh. The vLLM environment and results/env-vllm.json are those of D-019.
. "$HOME/pagedserve_work/env.sh"
cd "$PAGEDSERVE_WORK/paged_serve" || exit 1
ENGINE_PY="$PAGEDSERVE_WORK/venvs/pagedserve/bin/python"
VLLM_PY="$PAGEDSERVE_WORK/venvs/vllm/bin/python"
MODEL=models/Llama-3.2-1B-Instruct
NULL_PORT=18499
LATENCY_RATE=8
declare -A PROBE_RATE=([E]=64 [G]=32 [V64]=192 [V32]=128 [VE]=96)

log() { echo "== $(date -u +%FT%TZ) $*"; }

selftest() {
  "$ENGINE_PY" -m bench.gpu_selftest > "results/selftest-before-$1.json" 2>&1
  rc=$?; log "self-test before $1 exit $rc"
  [ $rc -eq 0 ] || { log "STAGE-FAILED self-test before $1: not measuring on this GPU"; exit 1; }
}

refuse_if_exists() {  # write-once outputs
  for f in "$@"; do
    [ -e "$f" ] && { log "STAGE-FAILED $f exists (write-once)"; exit 1; }
  done
}

raise_fd_limit() {
  # The in-flight ceiling of D-025 was probably a 1,024 open-file limit nobody had read. Raise the
  # soft limit before any server or generator starts; both inherit it, and the sweep records what
  # each process really sees.
  local hard target=65536
  hard=$(ulimit -Hn)
  if [ "$hard" != unlimited ] && [ "$hard" -lt "$target" ]; then target=$hard; fi
  ulimit -Sn "$target"
  log "open-file limit: soft $(ulimit -Sn), hard $(ulimit -Hn)"
}

shard_count() {  # half the cores, between 2 and 8 (D-030)
  local k=$(( $(nproc) / 2 ))
  [ "$k" -gt 8 ] && k=8
  [ "$k" -lt 2 ] && k=2
  echo "$k"
}

setup_session() {  # TAG
  raise_fd_limit
  tar xzf "$PAGEDSERVE_WORK/tmp/d025-code.tgz" -C "$PAGEDSERVE_WORK/paged_serve"
  cp "$PAGEDSERVE_WORK/tmp/COMMIT" COMMIT
  log "code installed, commit $(cat COMMIT); gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  nohup nice -n 10 "$ENGINE_PY" -m bench.machine_monitor --out "results/machine-monitor-$1.jsonl" \
      --interval 2 > "$PAGEDSERVE_WORK/monitor-$1.log" 2>&1 < /dev/null &
  sleep 6
  SHARDS=$(shard_count)
}

end_session() {  # TAG
  sleep 6
  pkill -TERM -u "$(whoami)" -f '^[^ ]*python -m bench.machine_monitor'
  sleep 4
  log "$1 session over"
}

step0() {  # TAG: the sharded generator against a server that does no work
  nohup nice -n 5 "$ENGINE_PY" -m bench.null_server --port "$NULL_PORT" --workers 4 \
      > "$PAGEDSERVE_WORK/null-server.log" 2>&1 < /dev/null &
  sleep 4
  local k good=""
  for k in "$SHARDS" $(( SHARDS * 2 )); do
    local out="results/d030-step0-$1-k$k.jsonl"
    "$ENGINE_PY" -m bench.loadgen run --url "http://127.0.0.1:$NULL_PORT" --model null \
        --rate 192 --duration 30 --vocab-size 100000 --seed 0 --warmup 1 --shards "$k" \
        --label "null-server-k$k" --server-env results/env-engine.json --out "$out"
    if "$ENGINE_PY" -m bench.matched_report generator "$out"; then good=$k; break; fi
  done
  pkill -TERM -u "$(whoami)" -f 'python -m bench.null_server'
  sleep 2
  if [ -z "$good" ]; then
    log "STEP0-FAILED: the generator is still the limit at $SHARDS and $(( SHARDS * 2 )) shards; the vLLM arms will be invalid or lower bounds and no ratio to vLLM may be quoted"
  else
    SHARDS=$good
    log "step 0 passed with $SHARDS shards"
  fi
}

arm_args() {  # ARM -> ARM_ARGS
  case "$1" in
    E)   ARM_ARGS=(--system engine --server-env results/env-engine.json --batch-sizes 64 --kv-cache-gb 8
                   --server-arg=--paged-attention-kernel --server-arg=--cuda-graphs) ;;
    G)   ARM_ARGS=(--system engine --server-env results/env-engine.json --batch-sizes 64 --kv-cache-gb 8) ;;
    V64) ARM_ARGS=(--system vllm --vllm-python "$VLLM_PY" --server-env results/env-vllm.json --batch-sizes 64) ;;
    V32) ARM_ARGS=(--system vllm --vllm-python "$VLLM_PY" --server-env results/env-vllm.json --batch-sizes 32) ;;
    VE)  ARM_ARGS=(--system vllm --vllm-python "$VLLM_PY" --server-env results/env-vllm.json --batch-sizes 64
                   --server-arg=--enforce-eager) ;;
    *)   log "unknown arm $1"; exit 1 ;;
  esac
}

run_arm() {  # ARM STAGE SEED RATE [INFIX]
  local arm=$1 stage=$2 seed=$3 rate=$4 infix=${5:-}
  arm_args "$arm"
  "$ENGINE_PY" -m bench.sweep --model "$MODEL" --rates "$rate" --seed "$seed" \
      --out "results/d030-$stage-$arm-rtx5090-${infix}seed$seed" \
      --loadgen-arg=--shards="$SHARDS" --resume "${ARM_ARGS[@]}"
  log "$arm $stage seed $seed at $rate req/s exit $?"
}

rates_report() {
  "$ENGINE_PY" -m bench.matched_report rates --results results --waive-audit
}

probe_arm() {  # ARM: one probe at seed 0, repeated once at twice the rate if it was not saturating
  local arm=$1 line again
  run_arm "$arm" probe 0 "${PROBE_RATE[$arm]}"
  line=$(rates_report | grep "^$arm:")
  log "probe $arm: $line"
  if echo "$line" | grep -q "repeat once at twice the rate"; then
    again=$(echo "$line" | sed -n 's/.*: \([0-9.]*\) req\/s$/\1/p')
    run_arm "$arm" probe 0 "$again" repeat-
    log "probe $arm repeated: $(rates_report | grep "^$arm:")"
  fi
}

main_rate_of() {  # ARM -> its offered rate from its probe (empty if there is none yet)
  rates_report | sed -n "s/^$1: ok: main offered rate \([0-9.]*\) req\/s\$/\1/p"
}
