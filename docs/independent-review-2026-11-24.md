# Independent review of pagedserve, and the response (a living document)

This file holds an independent review written by a separate, sceptical reviewer who recomputed
every table from the raw files. **The review itself is below, unedited** (its snapshot was commit
`0cff4dd` plus uncommitted work, on 2026-11-24). **This section is the project's response and is
updated as findings are resolved.** Where the response and the review differ, the review is what
was found and this section is what has been done since.

Status key: **done** (changed and committed), **in progress** (running or queued), **open** (not yet
started, or a decision the owner has not made).

Last updated: 2026-11-24.

| Finding | Summary | Status | Where / what |
| --- | --- | --- | --- |
| B1 | The 16 req/s cell is capped by the arrivals; "1.10x vLLM" understates a 4x or larger capacity gap | **done** | Pass 5 measured capacity above the arrival cap, four seeds paired: engine 14.2 req/s, vLLM 59.6, so vLLM is 4.2 times the engine (3.7 to 4.5) (`results/saturation-rtx5090.md`, D-019); README, STATUS and D-019 corrected |
| B2 | D-020's end-to-end criterion was unreachable | **done** | D-020 amended before any code: saturated throughput at 32 req/s offered, paired per-seed ratios interleaved in one session, interpretation table fixed in advance |
| B3a | bf16 scheduler divergences unexplained | **done** | Frozen-prompt scheduler evaluation: fp32 50 of 50 identical (35 preemptions included); bf16 35 of 50, reproducible, every divergence within one bf16 logit step in Hugging Face's own logits (D-019) |
| B3b | Prefix-cache hits never compared with an uncached run on real weights | **done**, one gap left | Shared-prefix scheduler evaluation (D-019): fp32 50 of 50 with 43,008 tokens served from the cache, and cache on equals cache off; bf16 differences are ties within one step. **Not covered:** preemption together with cache hits (that run did not preempt) |
| B4 | README and STATUS contradicted the data | **done** | README, STATUS, the setup-script header, the test count (382 passed on Windows) and the push state corrected; the GPU-era commits are now pushed, so CI runs on them |
| S1 | Prefill, the event loop and the interpreter lock were missing from the story | **done** | Offline run and thread counters recorded (D-019, D-021): the engine thread is 97 to 98% inside `scheduler.step()`, 2 to 3% routing, almost never idle. My first reading (the server layer loses two thirds) is withdrawn. The controlled follow-up on the real length mix (D-021): the workload doubles the step cost, the prefix cache costs under 1%, and the server adds a further 19 to 31% (untested cause) |
| S2a | vLLM run at defaults only | **done** | Eager control (two seeds): capacity 29 req/s against 59.6 with graphs, so graphs are worth about 2 times of vLLM's capacity and eager vLLM is still about 2 times the engine (D-019) |
| S2b | "Matching features" vLLM run, matched max batch, and a stronger baseline than static batching were not done | **open** | Needs a decision on scope; Hugging Face's own continuous batching is the obvious stronger baseline |
| S3 | Four seeds are workload draws; use paired per-seed ratios; run order not interleaved at run level | **done** for capacity | Pass 5 interleaved the systems within each seed and `bench/saturation_report.py` reports paired per-seed ratios; the earlier passes keep their original order |
| S4 | "Repeat versus original" evidence weaker than its headline | **open** | Reword the STATUS headline to lead with the seven informative saturated pairs and the one-directional Hugging Face drift |
| S5 | Prefix table: best-case workload, on-then-off order, unmonitored runs | **open** | Add the order caveat and a non-best-case workload |
| S6 | fp32 on this GPU is about 5.6 times less accurate than the CPU in the tail | partly **done**; cause **open** | `bench/fp32_probe.py`: with the test settings every single operation is as accurate as on the CPU (about 1e-6) and TF32 is not leaking in (it would show 3e-4); the tail error comes from accumulation over the 16 layers. A layer-by-layer comparison against float64 would locate it; not done |
| Part 3 | Not reproducible by a stranger | partly **done**; rest **open** | Pass 5 recorded weight file hashes, dependency freezes of both environments and hashes of the exact code that ran (`results/weights-sha256.txt`, `env-freeze-*.txt`, `box-code-hashes-pass5.txt`). Still open: a provider-agnostic script for one headline cell, per-result commit provenance, a pinned vLLM dependency closure beyond the freeze |
| Part 4a | Mutation-testing claims cannot be rerun | **open** | Soften the wording or add a script |
| Part 4b | Weak GPU assertions (top-1 above 0.8) | **open** | Tighten or relabel |
| Part 5 | D-020 critique (baseline, length mix, gates, confounds, self-attested pre-registration) | **done** | v2 criteria covered the baseline, gates and interpretation; the commits were pushed, which timestamps the plan. The review's length-mix point turned out to be decisive: profiled on the real mix the step is GPU-bound (graphs at most 1.07 to 1.14 times), so D-020 is parked, and D-025 (attention that reads the blocks in place) replaces it as the speed experiment. **Update 2026-11-24:** D-025 was built and tested on the GPU; correct in fp32, but it failed its fixed bf16 gate by one divergence (0.1875 against 0.125) also after the one allowed fix, so by the rule the claim was withheld. **Later the same day:** a diagnostic run (1.55 times saturated throughput) and D-026 (the kernel is as accurate as the gather path in bf16 against fp32, criteria post-hoc and written before the run) allow the figures to be quoted with both facts; D-020 is to be re-tested on the kernel path |
| Part 6 | Resume wording: use the saturated ratio; call the router a prototype validated on a simulated fleet; "paged attention" has no experiment of its own | **open** | Wording to be settled once pass 5 gives the engine's saturated rate |

---

# Independent review of pagedserve

Reviewer stance: sceptical, read-only. Everything below was recomputed from the raw files in
`results/` (scratch scripts outside the repo), not taken from the prose. Snapshot: HEAD `0cff4dd`
plus a working tree that **changed while I was reading it** (another session modified
`bench/profile_decode.py`, `bench/scheduler_eval.py`, `tests/test_scheduler_eval.py` and added
untracked files under `results/`: `fp64-truth-rtx5090.txt`, `gpu-tests-rtx5090.txt`,
`scheduler-eval-rtx5090-{bf16,fp32}.json`, `vllm-rtx5090-ceiling/`, monitor/audit files). I read the
untracked result files as raw data. None of them is cited by a committed doc yet except
`fp64-truth-rtx5090.txt`, which D-019 cites but a clean checkout does not contain.

The test-suite section (part 4) was done by a delegated read-only subagent; I spot-checked its
scheduler-eval findings against the raw JSON but not its other items. Its line numbers did not all
match the files, so I cite functions, not lines, for those.

---

## Summary verdict

The engineering discipline is unusually good: write-once results, a determinism self-test that
caught a faulty GPU, retractions kept in the record, seed repeats, machine audits. Every table I
recomputed reproduces to the digit. The weaknesses are in **what the headline metric measures**,
not in arithmetic:

1. The "delivered throughput at 16 req/s" metric is censored by the offered rate. It makes the
   engine look 1.10x from vLLM when the saturated gap is about 4x (blocker B1).
2. The pre-registered D-020 success criterion (c) cannot be met by any implementation (B2).
3. Every speed number is bf16 through the scheduler, and the first GPU check of that path shows
   11 to 16 of 50 prompts diverging from Hugging Face, unexplained (B3).
4. README and STATUS contradict the data (B4).

---

## Blockers

### B1. The headline comparison is censored; "1.10x vLLM" and "the gap is small" are wrong as a capacity statement

`request_throughput_rps = num_ok / (last completion - first send)` (`bench/metrics.py`). The
workload has a fixed request count (rate x 60 s = 960) with cumulative exponential gaps
(`bench/workload.py`), so the realised arrival span differs per seed (58.4 to 61.1 s).
A server that is not saturated therefore reports **960 / (arrival span + tail)**, a property of the
seed, not of the server.

Evidence, rate 16, per seed (engine = batch 64, vLLM = 32 seqs):

| seed | arrival-limited bound | engine | vLLM | HF b64 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16.41 | 13.52 (82%) | 16.24 (99%) | 8.44 |
| 1 | 15.72 | 15.13 (96%) | 15.60 (99%) | 9.44 |
| 2 | 15.69 | 14.23 (91%) | 15.50 (99%) | 8.98 |
| 3 | 15.96 | 14.51 (91%) | 15.86 (99%) | 9.28 |
| mean | 15.95 | 14.35 | 15.80 | 9.03 |

- vLLM sits at 99% of the arrival bound in every seed. Its pass-1 batch-32 and batch-64 runs gave
  the identical 16.24 req/s and 59.1 s duration, because only the arrivals mattered.
  The vLLM "spread" (2 to 6%) is arrival noise, not system noise.
- The engine is at 91 to 96% of the bound on seeds 1 to 3, so 16 req/s barely discriminates it from
  vLLM either. Engine seed 1 at batch 64 had TTFT p50 of 71 ms. In 3 of 4 seeds the batch-64
  median TTFT at 16 req/s was under 0.5 s (71, 367, 498 ms). The "mean of medians 1563 ms" quoted
  in the variation table is driven by seed 0 (5314 ms). "Overloaded at 16" is borderline for batch 64.
- The real gap, from the **untracked** `results/vllm-rtx5090-ceiling/`: vLLM saturates at about
  60.5 req/s (8.7k tokens/s) with 32 seqs and 82.6 req/s (11.9k tokens/s) with 64, both audited
  clean. The engine's best measured delivered rate is 14.35 req/s (2.06k tokens/s) and it was
  never offered more than 16. vLLM's saturated throughput is therefore about 4.2x (32 seqs) to
  5.8x (64 seqs) the engine's best *measured* rate. The engine's own ceiling is unmeasured: it is
  stepping 100% of the time at 16 req/s (see S1), so I expect it to be near 15 to 17, but that is
  inference.
- Where it appears: D-019 "vLLM at 16 req/s delivers 1.10 times ... but the gap is small",
  STATUS section 3, the change-log row "vLLM against the engine". "The gap is small" is false
  for capacity; D-019's "its ceiling is unknown" is now answerable.

Fix: report saturated tokens/s (closed-loop at fixed concurrency, or offered rate well above
capacity for all three systems) next to the fixed-rate latency table. Put the ceiling run in the
record. Replace "1.10x" with the saturated ratio. Run the engine at 24 and 32 req/s.

### B2. D-020 criterion (c) is unreachable (see part 5 for the full critique)

(c) requires mean delivered throughput at 16 req/s of at least 16.5 over four seeds. The
arrival-limited ceiling for those exact seeds is 15.95 on average (table above), and vLLM, which is
unsaturated, reaches 15.80. No implementation, including an infinitely fast one, can pass. The
experiment is guaranteed to end as a "partial/negative" result regardless of the CUDA-graph
outcome. Amend D-020 before any code exists (nothing is implemented, so an amendment is clean) and
say so in the entry.

### B3. Correctness of the benchmarked path is unestablished in bf16, and the prefix cache has never been checked on real weights

- All speed numbers are bf16 through `Scheduler` (batched, padded, explicit-mask attention).
  The only GPU check of that path is `results/scheduler-eval-rtx5090-bf16.json` (untracked): free-running
  64 tokens vs Hugging Face `generate`, 50 prompts. **39/50 identical** with no preemption (both
  prefix-cache settings), **34/50** with preemption (35 preemptions). First divergences are at tokens
  6, 7, 8, 11 and 16 for several prompts, which is early. fp32
  (`scheduler-eval-rtx5090-fp32.json`) is **50/50 in all three configurations**, including the
  preempting one.
- Reading: fp32 50/50 argues against a gross mask, paging or recompute bug. bf16 divergence is
  plausibly batch-shape rounding at near-ties (D-010 documents 1.7% exact ties). But the committed
  result files contain **no logit-gap evidence** that the divergent steps were near-ties (the
  uncommitted diff to `scheduler_eval.py` adds it; I did not see its output). Until then, "the speed
  numbers assume the engine served correct text" (D-004) is an assumption, and the data you now
  hold neither confirms nor refutes it for bf16. The plain-engine bf16 result (50/50, max logit diff
  0.0) says little about the scheduler path: it is bit-identical to Hugging Face because it calls
  the same kernels, which is not independent evidence.
- Prefix cache: in the scheduler eval, `served` mode had `prefix_hit_tokens = 0` (the prompts share
  no first block) and the preempting run had 64 hit tokens. The real-weights scheduler test uses
  prompts of at most 32 tokens with 16-token blocks. So no test, on any dtype or device, has compared
  generated text with prefix-cache **hits** on real weights, yet D-005 reports a 89 to 92% hit rate
  and a TTFT/throughput gain. The benchmark client records token counts only (`bench/client.py`),
  never content. I checked all 19,440 recorded engine requests on the 5090: 0 failures, every
  output has the requested length, so throughput is not flattered by short outputs. It does not show
  the tokens are right.

Fix: (1) a teacher-forced or logit-gap analysis of the 11 and 16 divergent prompts, recorded;
(2) a shared-prefix scheduler eval where the cache is hit, in fp32 and bf16; (3) record both result
files and a D-entry; (4) consider a cheap in-sweep check, such as hashing the output tokens of a
few fixed requests and comparing them across systems or with a reference.

### B4. The front-door docs contradict the data

- `README.md` line 5: "the engine has not been run on a GPU or benchmarked; the only measured
  numbers are for the Hugging Face baseline." The engine has been benchmarked on an RTX 5090. A
  hiring manager reads this first. The "Not yet benchmarked" sentence in its Engine server section is
  stale as well.
- `docs/STATUS.md`: the "Last updated" paragraph lists the ITL profile and the vLLM ceiling as not
  done (both done); section 3 is titled "The only real measurements so far" and opens with A6000
  numbers; section 6 items 1 and 2 say the scheduler has no GPU check and to "profile before trusting
  any throughput"; section 8 says D-001 to D-019 (D-020 exists); the setup script header says
  "UNTESTED until it has been run on a real instance" though it has been.
- STATUS says "Nothing is pushed". `git status -sb` shows `main...origin/main [ahead 10]`: the remote
  exists and earlier commits are there; the last 10 are not, so CI has never run on any of the
  GPU-era commits (STATUS item 6 acknowledges).
- STATUS test count (323 passed, 10 deselected = 333) does not match what the subagent collected
  (382 tests, 372 selected in CI). I did not run the suite, so I cannot say which is right; one is
  stale.

---

## Part 1. Claim-versus-data audit

Recomputed from the raw `*.jsonl` summary lines, manifests and JSON files. "OK" means the stated
number reproduces; the notes are where the wording is stronger than the data.

| # | Claim (source) | Recomputed | Verdict |
| ---: | --- | --- | --- |
| 1 | Engine b64 @16: 14.35 (13.52 to 15.13) (D-019) | 14.35 (13.52 to 15.13), n=4 | OK. Censored, see B1 |
| 2 | HF b64 @16: 9.03 (8.44 to 9.44) | same | OK |
| 3 | vLLM @16: 15.80 (15.50 to 16.24) | same | OK. 99% of arrival bound |
| 4 | vLLM @48: 47.67 (47.30 to 48.30), p99 TTFT 214 (98 to 420) | same | OK |
| 5 | Engine @4/8/12: 3.95, 7.64, 11.56 | same (b64, flagged runs excluded) | OK. Including flagged runs: 7.70 at 8 req/s; immaterial |
| 6 | Engine 1.6x (b64) to 1.7x (b32) HF | 14.35/9.03 = 1.59; 12.84/7.57 = 1.70 | OK, but vs HF that is also the engine's lower bound |
| 7 | vLLM 1.10x engine; "gap small" | 15.80/14.35 = 1.10 | Arithmetic OK; conclusion misleading. Saturated ratio 4.2x to 5.8x (B1) |
| 8 | HF wasted decode slots 48 to 63% in pass 1 | b32/b64, rates 4 to 16: 48.2 to 63.4% | OK |
| 9 | Engine median TTFT 37 ms, p99 under 90 ms through 8 req/s | 35 to 40 ms; p99 67 to 89 | OK |
| 10 | Engine "at its edge at 12": p99 321 ms (b64) | b64 mean of p99s 321 but range 82 to 996; b32 509 to 3322 | Mean quoted for one batch, range for the other; b64 vs b32 comparison is selective |
| 11 | Engine @16 b64 median TTFT 1.6 s "overloaded" | mean of medians 1563; medians 5314, 71, 498, 367 ms | Mean of medians hides that 3 of 4 seeds were not queued (B1) |
| 12 | Engine ITL "9 to 11 ms ... about 3x vLLM's 2 to 3" (STATUS, D-019) | ITL p50 at 2/4/8 req/s: 9.3 to 11.5 ms; at 8 req/s 11.2 to 11.5; at 16 b64 26.9 ms (seed 0), 21.9 to 24.7 (seeds 1 to 3); vLLM 2.3 to 2.6; TPOT agrees | Understated: about 4x at 8 req/s, 8 to 10x at 16 (S1) |
| 13 | D-003: step wall time "matches" pass-1 ITL so HTTP is not the main cost | Profile 7.6 to 10.2 ms; in-server ITL 9 to 11 at 2 to 4 req/s, 22 to 27 at 16 b64 | Holds only below about 4 req/s |
| 14 | Prefix table, engine 4/8/16 req/s throughput, TTFT p50/p99 (D-019) | 3.94/3.93, 7.81/7.75, 12.17/10.93; p50 32.5/41.1, 38.1/46.4; p99 66.7/78.3, 89.0/108.5 | OK (the "7.74" in the table is the 2-run unflagged mean; all-3 is 7.75) |
| 15 | Engine hit rate 89 to 92%, 0 with cache off | on: 0.894 to 0.920; off: 0.0 in all | OK |
| 16 | TTFT p50 fell 17 to 22% (engine), 24 to 39% (vLLM) | unrounded: engine 21.0% (4 req/s) and 18.0% (8); vLLM 23.2%, 30.1%, 37.3% | The quoted ranges come from rounded milliseconds (41/32, 46/38, 33/25, 36/22). Unrounded: 18 to 21% and 23 to 37%. Wider than the data |
| 17 | Engine @16 prefix: +11% throughput | 12.17 vs 10.93 = +11.3%; cache-on has 2 unflagged runs, off has 3; no overlap | OK, n=2 vs 3, overloaded regime, 3 seeds |
| 18 | Profile: 2.1x upper bound at b32 | 9.94/4.78 = 2.08 | OK |
| 19 | Attention + gather/copy 44% at b32, 56% at b64 | 26+18 = 44, 35+21 = 56 | OK |
| 20 | "GPU idle 52 to 61% up to batch 32" | 61, 57, 52% (b1, b8, b32); 39% at b64 | OK |
| 21 | Repeat vs original: 17 pairs, mean 1.3%, max 4.6% | engine 5 pairs 1.2%, HF 8 pairs 1.96%, vLLM 4 pairs 0.0%; overall 1.27%, max 4.6 | Arithmetic OK; informative content smaller (S4) |
| 22 | Monitor: 1,593 samples, 55 min, other CPU max 0.015, mean 0.0002, temp 27 to 66, power at most 456 W, throttle 178 | 1593; 55.4 min; 0.0150; 0.00020; 27 to 66; 455.9; bit 4 in 178 | OK |
| 23 | 29 monitored runs audit clean | 29 clean, 27 no-coverage in the table | OK |
| 24 | Six runs flagged contended; four in the sweeps | 6 flagged of 239 manifests; 4 in the main sweeps, 2 in pass 3 | OK. The "0 of 143 unflagged" denominator I could not reproduce (155 RTX 5090 manifests) |
| 25 | D-015: b16 4.02 (3.76 to 4.16), b32 5.26 (4.95 to 5.67), b64 5.31 (4.66 to 5.65) | identical in `results/variation.md` | OK |
| 26 | Spread "2 to 8% engine and vLLM where they keep up; 8 to 13% HF at every rate" | engine 4 to 8%, vLLM 2 to 6%; HF 8, 9, 13, 8, 16, 11% | HF at 16 req/s is 16% and 11%; the sentence is only true if overloaded cells are excluded |
| 27 | D-002/D-010: 217 of 12,800 steps exact ties; 112 steps HF breaks ties differently | 217 and 112 in `greedy-eval-a6000-bf16-ref-gpu.json` | OK |
| 28 | STATUS: greedy evaluation on the 5090 "50 of 50" | 50 prompts, 57 exact-tie steps, 30 reference self-mismatch steps | OK, but 50 prompts vs 200 elsewhere, and bit-identical because same kernels (B3) |
| 29 | fp32 5090: engine == HF GPU (0.0), 1.19e-3 from CPU | `gpu-tests-rtx5090.txt` (untracked): 1.190e-03, 0.0, 1.190e-03 | OK. See S6 |
| 30 | D-006 "reported both at defaults and with matching features" | Only defaults were run | Not delivered |
| 31 | D-019 cites `results/fp64-truth-rtx5090.txt` | File exists only untracked | A clean checkout lacks it |

Mismatches that flatter the project: rows 7, 12, 13, 16 (range slightly wide), 11, 21 (see S4).
No arithmetic errors found.

---

## Part 2. Statistical validity and fairness

### Should fix

**S1. The ITL / launch-overhead story is only true at low load, and prefill is about half the time at saturation (estimate).**
At 4 req/s the engine already runs a decode step continuously: batch-64 seed 1 recorded 6,375 decode
steps in a 63.5 s run (about 10 ms each) with a mean batch of 5. The step costs about 10 ms
regardless of batch, so "keeping up" means steps keep coming. At 16 req/s batch 64: 2,892 decode
steps, mean batch 47, over 63.4 s, so about 22 ms per decode step amortised. At the profiled
10 ms per step the decode steps account for about 30 s, leaving about 33 s for 960 one-at-a-time,
synchronising prefills (`Scheduler._admit` loops over waiting requests, each ends in
`int(logits.argmax())`). That is an inference from counters, not a measurement. The uncommitted
`offline_throughput` in `profile_decode.py` is exactly the right experiment to separate decode,
admit and HTTP. Run it and record it before claiming where the time goes. D-003's "both levers
matter" omits the third (prefill) and fourth (event loop and GIL contention between the HTTP
thread and the engine thread).

**S2. Fairness of the three-way comparison.**
- vLLM ran at defaults only (v0.34.0: CUDA graphs, chunked prefill, async scheduling, whatever
  else is default). D-006's "matching features" run was not done. A `--enforce-eager` run would
  attribute the gap to CUDA graphs directly and cheaply, and would test the D-003/D-020 hypothesis
  without writing any code.
- Max batch: engine headline batch 64, HF 64, vLLM `--max-num-seqs 32`. vLLM at 64 was run once per
  rate (seed 0) and its ceiling at 64 is higher (82.6 vs 60.5 req/s). Not matched where it matters.
- Output form: the engine and HF stream token ids; vLLM streams detokenized text (D-015 notes this
  slightly favours the id-streaming systems). Small, but unquantified.
- HF baseline is `generate` with a static batch and 20 ms window, tuned over batch size and window
  only. It is the weakest reasonable baseline (no length bucketing, no static cache or compile).
  Comparing to Hugging Face's own continuous batching, TGI or SGLang would be the question a systems
  reviewer asks. (I believe recent transformers releases ship a continuous-batching path; I did not
  check the version pinned here.)
- The same client machine runs the load generator and the server (24 cores, shared fleet host).
  The generator never flagged saturation (max p99 lag 10 ms over 239 runs), so this is fine, and
  at 128 req/s the ceiling runs do flag saturation (p99 lag 0.16 to 0.20 s) and are also
  deadline-truncated (4,563 of 7,680 ok at b32), so the 128 req/s cells are not clean throughput.

**S3. Statistics.**
- Four seeds, min-max ranges. The seeds are workload draws (lengths and arrivals), so the spread is
  mostly workload variation, not system noise, and the runs are naturally paired by seed across
  systems. A per-seed ratio (engine/HF, engine/vLLM) with its own range would be a stronger and
  simpler statement than comparing unpaired mean-min-max.
- Complete separation of 4 vs 4 is a permutation p of 2/70 = 0.029. Fine for engine vs HF; the
  engine vs vLLM separation is an artifact of censoring (B1).
- Run order: per seed the systems run engine, HF, vLLM in blocks, rates ascending. Clock behaviour
  differs by block (min clock ratio about 0.84 for HF repeats vs 0.92 for engine; power-capped samples
  11% of the monitored window). Not interleaved at run level; effect small versus the headline gaps
  but unquantified. vLLM is always last in each seed.
- Excluding contended runs: immaterial. Engine b32 @8: 7.69 unflagged vs 7.74 all; b32 @16: 12.84 vs
  12.78. The flag is a weak signal (S4), which the doc already says.
- 60 s of arrivals plus drain, fresh server per run, 3 warm-up requests. For the engine this includes
  first-use effects (allocator growth, kernel selection) inside the measured window. TTFT p50 is
  unaffected; saturated throughput might be slightly.
- "Poisson" arrivals with a fixed request count are conditioned on N, so the realised rate is 15.7
  to 16.4 for a nominal 16. State it.
- Single GPU, single day (all passes within about 5 hours), single driver stack (CUDA 13.2 runtime on a
  driver advertising 13.0, a configuration D-016 itself calls suspect).

**S4. The "repeat vs original" evidence is weaker than "17 pairs agree to 1.3%".**
vLLM's four pairs (0.0%) are arrival-determined and prove nothing about the machine; the engine and HF
@8 req/s pairs are below saturation where throughput equals the offered rate by construction. The
informative pairs are the saturated ones: engine @16 (+0.7, -0.5, +1.7; one more excluded as flagged)
and HF @16 (+4.6, +1.9, +4.1, +1.5). That is 7 pairs, and all four HF repeats are faster by 1.5 to
4.6%, a one-directional drift (a 1 in 16 coin-flip) that the repeat's mean-absolute summary hides.
The docs do say "informative pairs are the saturated cells"; the headline sentence in STATUS does not.
Also, "other GPU processes: 0" depends on `nvidia-smi` seeing other users' processes; on a shared
containerised fleet that may be blind. I could not verify this.

**S5. Prefix-cache table.** See rows 14 to 17 and B3. Additionally: 27 of 36 runs are unmonitored;
the "on" vs "off" runs are always in on-then-off order; the workload (8 prefixes x 1024 tokens, skew 1,
about 93% maximum possible hit rate) is the best case for the feature, and the gain in the regime
people care about (not overloaded) is 8 to 14 ms. The honest summary is "works, small absolute gain
on this GPU, correctness on real weights unchecked".

**S6. The fp32 GPU contract was changed after it failed, and the replacement is partly vacuous.**
The 1e-3 bound against the CPU failed on the 5090 (1.19e-3); the contract was then changed to
"equals HF on the same GPU within 1e-4" and "no further from the CPU than HF's GPU run (within 5%)".
D-019 documents this openly, which is good. But the engine equals HF-GPU exactly (0.0), so the second
check is implied by the first, and the first is true because both call the same cuBLAS and SDPA
kernels, so it cannot detect a bug that is not in the engine's glue. The independent evidence is the
fp64 truth report (untracked): max error against float64 of HF fp32 on CPU is 1.8e-4, but HF and engine
fp32 on this GPU are **1.012e-3**, about 5.6x worse, and 2.4x worse than the A6000's 4.2e-4 (D-002).
"fp32" on this card is not behaving like full-precision fp32 (TF32 or an emulated path inside cuBLAS on
this CUDA build is a candidate). I did not test this. It deserves a one-line check
(`torch.backends.cuda.matmul.allow_tf32`, `torch.get_float32_matmul_precision()`, cuBLAS emulation
settings) because it bears directly on how much to trust "fp32 matches".

---

## Part 3. Reproducibility

Could a stranger reproduce one headline number from a clean checkout? **Not as written.** Missing or
broken:

1. Hardware: an Oxmiq Capsule RTX 5090 machine reached through an internal CLI. The scripts
   (`deploy/gpu_pass*.sh`) source `$HOME/pagedserve_work/env.sh`, which is not in the repo, and
   hard-code the work layout. A stranger would need to rewrite them. There is no provider-agnostic
   "reproduce the 16 req/s table" command and no Makefile target.
2. Weights: Llama-3.2-1B-Instruct is gated; the repo tells you to download it by hand. No revision
   or file hash is recorded in any result or environment file (I searched `env-*.json`). A
   different revision could change tokens.
3. vLLM environment: `pip install vllm==0.34.0` with transitive dependencies unpinned; only
   `pip freeze | grep vllm|torch` is printed, and `env-vllm.json` records top-level packages. The vLLM
   result cannot be recreated bit for bit in its dependency closure.
4. Provenance: every env capture says `git.commit: null` because the box ran a `git archive`. I
   verified `git diff b101006 HEAD -- engine bench/sweep.py bench/loadgen.py bench/client.py
   bench/workload.py bench/baselines` is empty, so the code is recoverable, but only because you
   told me it was b101006; the result files do not say.
5. Driver stack: CUDA 13.2 runtime on a driver advertising 13.0, via an `ALLOW_OLDER_DRIVER=1`
   override in a setup script whose header still says untested. A rented 5090 on a normal driver would
   differ in an unknown way.
6. The commands that produce `results/variation-*-rtx5090.md` and which directories feed each are not
   written down (the runbook line shows a generic form; the real inputs included the p1, seed0-extra
   and seed 1 to 3 directories). I reproduced the tables by gathering those directories.
7. Machine-specific caches, 20-minute setup and the cleanup duty are in prose, not scripts.
8. Pinned versions (`torch 2.15.0+cu132`, `transformers 5.23.0`, `vllm 0.34.0`) postdate my knowledge;
   I could not verify them and did not try to.
9. Results are in git, but 10 commits are unpushed and three evidence files cited by the docs are untracked.

What does reproduce: every table from the committed raw files with a few lines of Python.

---

## Part 4. Correctness evidence (what the tests prove)

From the subagent, spot-checked where marked (V = I verified in the raw data):

- CI runs ruff, mypy and `pytest -m "not gpu and not real_model"`. All correctness tests that run
  in CI are tiny random-weight fp32 CPU models (2 layers, hidden 64, vocab 128, sequences about 40
  tokens, block size 4). Real-weights (5 CPU tests) and GPU (3) tests, `greedy_eval` and
  `scheduler_eval` run by hand and skip silently when weights are absent.
- Proven: forward pass vs Hugging Face on tiny and, in fp32 on CPU, on the real model (bit-identical,
  50 prompts and a 2048-token prompt); paged vs contiguous bitwise on tiny; batched scheduler =
  single-request greedy on tiny; scheduler = HF `generate` on 16 real prompts in fp32 CPU including 9
  preemptions; (V) scheduler = HF on 50/50 prompts in fp32 on the 5090 in all three configs.
- Not proven: scheduler in bf16 on GPU (39/50 and 34/50, B3); prefix cache hits on real weights in any
  dtype; anything paged or batched beyond about 56 tokens on real weights (the 2048-token check is a
  plain forward pass).
- Weak assertions: GPU and real-model bf16 tests only require top-1 agreement above 0.8, so a bug
  costing up to 20% of tokens passes. `test_model.py` only asserts bf16 logits are finite plus fp32
  `inv_freq`.
- The mutation claims ("deliberately broken variants fail", "six", "seven of eight", "37") are not
  backed by any script in the repo; the only in-repo mutant is a q_proj x1.5 test. They were done by
  hand and cannot be rerun. A reviewer cannot tell a claim of 37 from a claim of 3.
- The stale-block test cannot detect a stale-block leak because the free list is FIFO and the second
  run gets never-used blocks (low risk since eviction and preemption tests recycle).
- `scheduler_eval` has no pass/fail threshold and all 50 prompts arrive at once (no staggered arrivals,
  no aborts). Its reference is independent HF `generate`, which is a strength.
- Engine reading: decode mask (`PagedBatchCache.attention_mask`, column <= length) is correct; padded
  table slots point at block 0 and are masked (a NaN or Inf in block 0 would turn the masked
  products into NaN, which is a latent hazard once CUDA graphs add padding rows); prefix hash chains
  parent hash and token ids; EOS ignored by the benchmark client; sampling is argmax only.

---

## Part 5. D-020, the pre-registered CUDA-graph experiment

What is good: criteria written before code; an opt-in flag so old results stay reproducible; a
stopping rule that forbids moving the criteria; a negative result is allowed.

What is wrong or insufficient:

1. **(c) is unreachable (B2).** 16.5 > 15.95 (the arrival ceiling for seeds 0 to 3). "Lowest seed above
   15.1" demands 96% of the ceiling on seed 1 and more than that on others. Replace (c) with a metric that
   is not censored: saturated tokens/s at a rate above capacity (say 24 and 32 req/s) or closed-loop at
   fixed concurrency, with per-seed paired ratios against the baseline measured in the same session.
2. **The baseline is a stored number from another session.** 9.94 ms (b32) came from one run of 40
   steps; its own range was 9.79 to 18.17 ms at batch 32. A/B must be interleaved in one session with
   the same monitor. A 30% bar is large enough to survive noise; a stored baseline is not.
3. **(b) is measured on the friendliest shape.** `profile_decode` uses 32 sequences of exactly
   256 tokens, no prefill in the measured steps, one batch size per bucket. The workload has lognormal
   prompts up to 2048 tokens and outputs to 512. Padded width follows the longest sequence, so bucketed
   graphs will pad more and gather more. Measure the step time on the real length mix and at mean batch
   sizes the server actually sees (5 at 4 req/s, 13 at 8, 24 at 12, 47 at 16).
4. **It ignores prefill and the HTTP/GIL path (S1).** If prefill and event-loop costs are about half of a
   saturated run, graphs on decode can improve end-to-end throughput by at most the decode share. The
   hypothesis should state an expected end-to-end gain and the measured decode share before
   implementation, so a "partial" result is interpretable.
5. **Correctness gate (a) is unfalsifiable at the margin in bf16.** "Identical counts no worse than the
   default path": the default is 39/50 and 34/50 in bf16. A correct graph path can flip a different
   handful of near-ties in either direction; an incorrect one with small errors can pass. Use instead:
   (i) fp32 scheduler eval 50/50 in all three configs, (ii) bit-equality of eager vs graph logits on
   identical (unpadded) inputs, since replay uses the same kernels, (iii) a bounded difference with
   padding rows, (iv) teacher-forced top-1 agreement with logit-gap accounting in bf16. The gate also
   needs a stated sample size (50 prompts has a 95% lower bound of 93% at 50/50).
6. **Missing gates for the failure modes graphs introduce:** padding rows writing to a scratch block
   while a live sequence uses that block; stale static buffers across requests; bucket switches mid
   sequence; preemption plus prefix-cache hits under graphs; memory (graph pools shrinking the KV
   pool and so changing preemption counts); capture under the server's threading model.
7. **Confounds in what could be learned.** The change bundles graphs, static buffers and probably a
   changed gather. A positive result would not isolate graphs. The cheapest control, vLLM
   `--enforce-eager` vs default, is not part of the plan and would bound how much graphs can matter at
   all on this stack.
8. **"Pre-registered" is self-attested.** The commit lives only in an unpushed local branch and can be
   rewritten. Push the commit (or a signed tag) to the remote, or send the hash somewhere with a
   timestamp, before implementing.
9. **Time-box vs outcome space.** "Partial result" is allowed when (a) passes and (b) or (c) fails, but
   there is no pre-stated interpretation (for example, (b) passes and (c) fails means "decode is not
   the bottleneck at saturation", which is a result worth stating in advance).

---

## Part 6. Resume readiness

### Overclaimed or should be cut

- "Within 10% of vLLM" or any phrasing derived from 14.4 vs 15.8. Use the saturated ratio.
- The prefix-aware router: evidence is a mock backend with uncalibrated placeholder timing, three
  seeds, run on a Windows laptop, never on a real replica. Say "router prototype, validated on a
  simulated fleet" or cut. Do not put hit-rate numbers from it on a resume.
- "Paged attention" as a performance feature: the implementation gathers every sequence's blocks into a
  padded contiguous tensor each layer each step; D-003 has no measurement of what paging buys (memory
  efficiency or concurrent sequences at fixed memory). The headline feature of the project has no
  experiment attached. Either measure it or describe it as a "paged KV cache with gather-based
  attention".
- "Continuous batching speedup" is only measured against a weak static-batching baseline, with all
  other differences between the stacks included.
- "Pre-registered experiment" before it is pushed or its criterion is fixed.
- "Four seeds" and "machine audit" as proof of rigor are fine, but the seeds are workload draws.

### Keep

- From-scratch Llama forward pass bit-identical to Hugging Face in fp32 on CPU, and scheduler-versus-HF
  fp32 50/50 on GPU with preemption.
- A determinism self-test that caught a faulty GPU and led to retracting results (D-016, D-002
  correction). This is the strongest single story in the repository.
- Honest negative statements: engine slower than vLLM, retractions, correcting the batch-64 conclusion.
- Profile-before-optimise with a quantified upper bound (2.1x at b32).
- A reusable open-loop load generator with write-once results and a machine monitor.

### Strongest honest one paragraph

> pagedserve is a from-scratch Llama-3.2-1B inference engine in PyTorch: a block-allocated paged KV
> cache with hash-chained prefix caching, an iteration-level continuous-batching scheduler with
> preemption by recompute, and a streaming OpenAI-style server, with a benchmark harness (open-loop
> load generator, write-once results, GPU determinism self-test, machine-contention monitor). In fp32
> it is token-identical to Hugging Face (50/50 prompts through the scheduler on an RTX 5090, including
> forced preemption). On one RTX 5090 it delivers about 1.6x the throughput of a tuned Hugging Face
> static-batching server, but it is roughly 4x or more behind vLLM at saturation. A decode-step
> profile showed the engine is launch-bound (GPU idle 52% at batch 32), which motivated a
> CUDA-graph experiment. The harness also caught a faulty GPU that returned different results for
> identical inputs.

(Edit "roughly 4x or more" once the engine's own ceiling is measured.)

### Five interview questions the docs cannot yet answer

1. "What is the engine's saturated throughput, and where does the time go?" The engine was never
   offered more than 16 req/s, and the decode / prefill / HTTP split is not recorded.
2. "Is the 39/50 and 34/50 bf16 agreement rounding or a bug?" No logit-gap analysis is recorded for
   the divergent prompts.
3. "Does the prefix cache return the same text on real weights, and what does it buy outside the best-
   case 8 x 1024 workload?" Cache hits have never been compared with an uncached run on real weights.
4. "What does paging buy in your engine?" No memory, fragmentation or concurrency measurement; decode
   gathers a padded contiguous copy each step, so what is "paged" about the speed?
5. "How do you know the gap to vLLM is CUDA graphs, and why is your baseline static batching rather
   than a continuous-batching one?" No `--enforce-eager`, chunked-prefill-off or matched-feature vLLM
   run, and no stronger baseline.

---

## What I could not verify

- Whether the test suite passes (I did not run it; counts in STATUS and the subagent's collection
  disagree).
- Anything about the Capsule machine itself, other users' GPU activity (the monitor may not see other
  containers' processes), or the `git archive` contents on the box.
- Package versions and the existence of the pinned releases; whether recent transformers has a
  continuous-batching path; vLLM 0.34 defaults beyond what the code passes.
- The cause of the 1.0e-3 fp32 GPU error (S6) and of the bf16 divergences (B3).
- The denominator "143" in the sweep-flag statistic.
- The content of uncommitted changes in progress (logit-gap analysis, `offline_throughput`).

---

## The five highest-value fixes

1. **Replace the censored headline with a saturation measurement.** Record the vLLM ceiling run, run
   the engine at 24 and 32 req/s, report tokens/s at saturation and paired per-seed ratios; delete
   "1.10x" and "the gap is small" (B1).
2. **Amend D-020 before writing any code** (criterion (c), interleaved A/B baseline, real length mix,
   fp32 bit-equality and padding gates, push the commit) and add a `vllm --enforce-eager` run as the
   cheapest test of the hypothesis (B2, part 5).
3. **Close the bf16 and prefix-cache correctness gap:** record the scheduler evals, add the logit-gap
   analysis of the 11 and 16 divergent prompts, and add a shared-prefix scheduler eval that actually
   hits the cache in fp32 and bf16 (B3).
4. **Bring the docs into line with the data and commit the evidence:** fix README line 5 and the stale
   STATUS sections, commit the untracked result files that docs cite, push so CI runs, and replace the
   hand-run mutation claims with a script or soften them (B4, part 4).
5. **Make one number reproducible from a clean checkout:** a provider-agnostic script and doc for the
   engine/HF/vLLM 16 req/s cell, with the weights revision hash, full vLLM freeze, driver and CUDA
   stack and the code commit recorded in every result (part 3). Check TF32 or emulation on the 5090
   while there (S6).
