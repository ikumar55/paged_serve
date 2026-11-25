#!/usr/bin/env bash
# D-030 session A: the headline. Step 0 (the sharded generator against a null server), one probe each
# for the engine with the kernel and graphs (E), the engine's gather path (G) and vLLM with 64
# sequences (V64), then their main runs at 1.5 times each arm's own probed capacity, four seeds,
# interleaved within each seed in a rotating order, the self-test before every seed. About 1.5 hours.
# Not run yet (the tools were built 2026-11-25; the machine was not available). Set WITH_D029=1 if the
# engine includes batched prefill, for the report's prediction ranges.
. "$(dirname "$0")/gpu_d030_lib.sh"
TAG=d030-a
refuse_if_exists "results/machine-monitor-$TAG.jsonl" results/d030-report-a.md

setup_session $TAG
selftest $TAG-start
step0 a

ARMS=(E G V64)
for arm in "${ARMS[@]}"; do probe_arm "$arm"; done
declare -A RATE
for arm in "${ARMS[@]}"; do
  RATE[$arm]=$(main_rate_of "$arm")
  [ -n "${RATE[$arm]}" ] || { log "STAGE-FAILED no usable probe for $arm; see the probe lines above"; end_session $TAG; exit 1; }
  log "main offered rate $arm: ${RATE[$arm]} req/s"
done

for seed in 0 1 2 3; do
  selftest "$TAG-s$seed"
  for i in 0 1 2; do
    arm=${ARMS[$(( (i + seed) % 3 ))]}
    run_arm "$arm" main "$seed" "${RATE[$arm]}"
  done
done
selftest $TAG-end

end_session $TAG
"$ENGINE_PY" -m bench.matched_report report --results results ${WITH_D029:+--with-d029} \
    --monitor "results/machine-monitor-$TAG.jsonl" --out results/d030-report-a.md
log "D030-A-DONE"
