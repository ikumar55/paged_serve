# Roadmap: planned scope against what is done

The original project plan (a gitignored `PROJECT_PLAN*.md`) was lost in a laptop reset and could
not be recovered from the repository, its history, the owner's OneDrive or the earlier session
transcripts. This roadmap is **reconstructed** from `docs/STATUS.md`, `docs/decisions.md` (D-011
refers to a "Phase 2 gate" at the end of Week 4 and a six-week plan) and the owner's local notes.
Where the original plan had items this list does not, they are missing; add them. A living
document: update the status column as work lands.

Last updated: 2026-11-24 (evening).

## How this roadmap is used (working rules)

1. **One item at a time.** Exactly one row is "current". Nothing else is built or measured until it
   is finished, recorded and committed, or explicitly parked here with a reason.
2. **Plan before building.** An item that needs an experiment gets its hypothesis, design,
   success criteria and interpretation written in `docs/decisions.md` first (the D-number is in the
   table). The plan can be amended only before any result exists, and the amendment is dated.
3. **Finish means recorded.** An item is done when its result is in `results/`, written up in the
   decision log and STATUS, its row below is updated, and it is committed. The review tracker
   (`docs/independent-review-2026-11-24.md`) is updated in the same step if a finding moves.
4. **Start every session by reading this file**, then work on the current item.
5. Where the data contradicts a claim, fix the claim first.

**Current item: 4 (speed work), step 5: D-028, where the saturated engine thread's time goes with
the kernel and graphs** (planned with predictions and reading fixed in advance; the accounting, the
report tool and the session script are built and tested on the CPU; **the measurement has not been
run: the GPU machine is off limits until later on 2026-11-24**). The same machine session will also
run the D-027 replay-count rerun (the graph path's accuracy result is in, but the result file does
not show that the graphs were really replayed with padding).

Done and recorded, in order: step 1 (pass 5: capacity, attribution, eager control, prefix-cache
correctness); step 2 (the D-021 follow-up: the real length mix doubles the step cost, the cache
costs under 1%, the server adds 19 to 31%); step 3 (the real-mix profile: the saturated gather
step was GPU-bound, attention and gather 84% of GPU time); **step 4, the speed levers:**
- **D-025, decode attention that reads the blocks in place** (Triton, opt-in
  `--paged-attention-kernel`): correct in fp32 (50 of 50 identical to Hugging Face, preemptions and
  cache hits included); **its fixed bf16 gate failed by one near-tie** (0.1875 against 0.125) and
  stayed failed after the one allowed fix. **D-026** (post-hoc criteria, written before the run)
  then showed it is as accurate as the gather path in bf16 against fp32. Labelled diagnostic: the
  real-mix step 52% shorter, saturated throughput **1.55 times** the gather path's.
- **D-020, CUDA graphs on the kernel path** (opt-in `--cuda-graphs`): capture works, fp32 outputs
  are hash-identical to the eager kernel's; **its strict bf16 gate (identical tokens) failed**
  (padding changes the GEMM shape). **D-027** (post-hoc, written first) showed the graph path's
  bf16 accuracy equals the eager kernel's. Labelled diagnostic: a further **1.40 times**.
- Together about 2.1 times the gather path (13.97 to 28.98 req/s); against vLLM of an
  earlier session (only a rough comparison) the gap is about 2.1 times with vLLM at 32 sequences
  (59.6 req/s) but **at least about 2.85 times with vLLM at 64 sequences, the engine's own setting**
  (82.6 req/s, a lower bound; found by the second review, D-019 already had the figure). Quote these
  figures only with both facts: the strict gates failed, and the accuracy standards were chosen
  after the fact (D-025, D-020 entries).

After D-028: if admission (the one-at-a-time prefills) is at least 20% of the step, batched
prefill is D-029 (plan first); otherwise the server layer. The remaining order is in **Plan from
here** below (it replaces the older order "then items 5, 6, 8, 7, 2, 9, 10").

## Plan from here (proposed 2026-11-24 evening after the second review; the owner confirms or changes it)

Aim: good results across the original breadth (engine, baselines, speed, router, paging,
deployment), with Docker and Kubernetes included, **without dragging on**. The second review
(`docs/second-review-2026-11-24.md`, tracker S2-1 to S2-18) is folded in. One item at a time inside
each phase; every experiment keeps the rule of a written plan before any code or run. Time boxes
are estimates, not promises.

1. **Phase 1, one machine session (about 1 hour of machine time, when the machine is available):**
   D-028 (amended to also run at 48 req/s offered, S2-6) and the D-027 rerun that records the
   graphs' replay counts (S2-8). Finish with the usual write-up, then decide D-029 from the
   admission share.
2. **Phase 2, no machine needed (about 1.5 days):**
   - **Credibility sweep (item 12):** push and get CI green (needs the owner's say-so); write the
     commit id into every result manifest and environment capture (S2-11); find the cause of the
     gather arm's ConnectErrors from the existing records (S2-3); cut the README to a short summary
     with one table and the three caveats (S2-9); a one-page summary table at the top of
     `decisions.md` (S2-16); build and CPU-test a sensitive bf16 accuracy tool, KL or perplexity on
     real text against fp32 and Hugging Face bf16 (S2-7).
   - **Deployment (item 7):** Dockerfiles for the engine server, the router and the load generator,
     and Kubernetes manifests (a Deployment with a GPU resource request, a Service, a router
     Deployment with the replicas as endpoints, config, probes on `/health`); validated offline
     with a manifest linter and a dry run, **marked untested on a cluster** unless a Docker host or
     a cluster is found.
3. **Phase 3, machine sessions (about 3 or 4):** **batched prefill (D-029) first if D-028 triggers
   it**, so that the next comparison measures the final engine; then the **matched vLLM comparison
   (item 11, D-030)**: same session, vLLM at 32 and 64 sequences, every system offered at least 1.5
   times its capacity, a load generator that is not the bottleneck, interleaved by seed, and a
   length-bucketed gather arm as a fairer own-baseline (S2-1, S2-2, S2-4, S2-5); then **D-023 what
   paging buys (item 8)** (S2-13); then **the router on real replicas (item 5, D-022)** with the
   replicas as processes on the one GPU (S2-14); the sensitive bf16 accuracy run on the tool from
   phase 2; the **kernel microbenchmark (item 13, D-031)**, GB/s against the card's peak and a
   comparison with PyTorch's own attention (and FlashInfer or flash-attn if they install), with
   split-KV and autotuning tried, only if time remains (S2-12).
4. **Phase 4, stretch:** the calibrated simulator (item 6, D-024) and generality on a second model
   or GPU class with real text (item 14, D-032) (S2-15).

**Cut line: after phase 3 the project is complete enough to present.** Phase 4 is done only if time
remains. If the machine stays unavailable, phase 2 can be done first.

| # | Planned item | Where it is decided | Status | What remains |
| ---: | --- | --- | --- | --- |
| 1 | Engine: paged KV cache, continuous batching with preemption, prefix cache, OpenAI-style server | D-002 to D-005 | **Done**, measured on one RTX 5090 | Stretch goals: chunked prefill (the fused paged-attention kernel is built, D-025, opt-in and unclaimed) |
| 2 | Baselines: Hugging Face static batching and vLLM | D-006, D-015 | **Done** (vLLM at defaults) | A stronger baseline than static batching; vLLM matching-features run (the eager control is in pass 5); a second GPU class |
| 3 | Benchmark harness and honest results: load generator, write-once results, determinism self-test, machine monitor | D-014 to D-016, D-019 | **Done** | Keep it current |
| 4 | Speed work, only after profiling | D-003, D-020, D-021, D-025, D-026, D-027, D-028 | **Current.** Kernel (D-025) and CUDA graphs (D-020) built and tested; strict gates failed in bf16, accuracy standards D-026 and D-027 met (post-hoc); diagnostics 1.55 times then 1.40 times, about 2.1 times the gather path in total; D-028 planned | D-028 measurement and the D-027 replay-count rerun (waiting for the machine); batched prefill (D-029) if admission is at least 20% of the step |
| 5 | Router with prefix-cache-aware policies, validated on **real replicas** (the Phase 2 gate) | D-007, D-011, **D-022** | Built; compared only on a mock fleet. **Pre-registered (D-022); harness not yet changed; nothing measured** | Add a real-backend option to `bench/router_compare.py`, then run D-022 |
| 6 | One extension: a calibrated serving simulator (chosen) or speculative decoding (declined) | D-011, **D-024** | **Pre-registered (D-024); not started** | Measure step and prefill times, fit the timing model, validate on held-out cells |
| 7 | Deployment: Dockerfiles and Kubernetes manifests | STATUS | Not written; **scheduled in phase 2 of Plan from here** | Write them; validate offline with a manifest linter and a dry run; mark untested on a cluster if Docker or a cluster is unavailable (Docker is not installed on the laptop) |
| 8 | What paging buys (raised by the independent review) | review, **D-023** | **Pre-registered (D-023); no run yet** | Run the block-size experiment at a binding memory budget |
| 9 | Review findings | `docs/independent-review-2026-11-24.md`, `docs/second-review-2026-11-24.md` | Tracked there | Follow their status tables |
| 10 | Cleanup of the shared benchmark machine | STATUS section 6, item 13 | Pending | Delete `~/pagedserve_work` after the last GPU work, and confirm the home directory is back to baseline |
| 11 | Matched same-session vLLM comparison (second review S2-1, S2-2, S2-4, S2-5) | second review, **D-030 (to plan)** | Not planned yet | vLLM at 32 and 64 sequences, every system offered at least 1.5 times capacity, a load generator that is not the bottleneck, a length-bucketed gather arm; interleaved by seed in one session; after D-029 if that is triggered |
| 12 | Credibility sweep (S2-3, S2-7, S2-9, S2-10, S2-11, S2-16, S2-18) | second review | Not started; phase 2, no machine needed | Push and CI, commit ids in results, ConnectError cause, README cut, decision-log summary table, a sensitive bf16 accuracy tool (KL or perplexity) |
| 13 | Kernel microbenchmark (S2-12) | second review, **D-031 (to plan)** | Not planned yet | GB/s against the card's peak, comparison with PyTorch attention (and FlashInfer or flash-attn if they install), split-KV and autotuning tried; only if time remains |
| 14 | Generality on a second model or GPU class with real text (S2-15) | second review, **D-032 (to plan)** | Not planned yet; stretch, phase 4 | Repeat the kernel-plus-graphs comparison on one more model, EOS-terminated lengths |
