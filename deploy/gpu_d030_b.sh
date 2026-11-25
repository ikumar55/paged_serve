#!/usr/bin/env bash
# D-030 session B: the controls and part 2. Step 0 again, one probe each for vLLM with 32 sequences
# (V32) and vLLM with 64 sequences eager (VE), their main runs, then the latency cell: E, G and V64
# at a common 8 req/s, four seeds. The final report covers both sessions (it reads session A's results
# and monitor file from this machine). About 1 hour. Not run yet. Set WITH_D029=1 as in session A.
. "$(dirname "$0")/gpu_d030_lib.sh"
TAG=d030-b
refuse_if_exists "results/machine-monitor-$TAG.jsonl" results/d030-report-b.md

setup_session $TAG
selftest $TAG-start
step0 b

for arm in V32 VE; do probe_arm "$arm"; done
declare -A RATE
for arm in V32 VE; do
  RATE[$arm]=$(main_rate_of "$arm")
  [ -n "${RATE[$arm]}" ] || { log "STAGE-FAILED no usable probe for $arm; see the probe lines above"; end_session $TAG; exit 1; }
  log "main offered rate $arm: ${RATE[$arm]} req/s"
done

for seed in 0 1 2 3; do
  selftest "$TAG-s$seed"
  for i in 0 1; do
    arm=$( [ $(( (i + seed) % 2 )) -eq 0 ] && echo V32 || echo VE )
    run_arm "$arm" main "$seed" "${RATE[$arm]}"
  done
done
for seed in 0 1 2 3; do
  selftest "$TAG-lat-s$seed"
  for i in 0 1 2; do
    arms=(E G V64)
    arm=${arms[$(( (i + seed) % 3 ))]}
    run_arm "$arm" lat "$seed" "$LATENCY_RATE"
  done
done
selftest $TAG-end

end_session $TAG
"$ENGINE_PY" -m bench.matched_report report --results results ${WITH_D029:+--with-d029} \
    --monitor results/machine-monitor-d030-a.jsonl "results/machine-monitor-$TAG.jsonl" \
    --out results/d030-report-b.md
log "D030-B-DONE"
