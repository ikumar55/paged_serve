# Roadmap: planned scope against what is done

The original project plan (a gitignored `PROJECT_PLAN*.md`) was lost in a laptop reset and could
not be recovered from the repository, its history, the owner's OneDrive or the earlier session
transcripts. This roadmap is **reconstructed** from `docs/STATUS.md`, `docs/decisions.md` (D-011
refers to a "Phase 2 gate" at the end of Week 4 and a six-week plan) and the owner's local notes.
Where the original plan had items this list does not, they are missing; add them. A living
document: update the status column as work lands.

Last updated: 2026-11-24.

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

**Current item: 4 (speed work), step 1: pass 5 and the server-loss attribution (D-021).**
Order after that: 4 (run D-020 or the server fix) then 5, 6, 8, 7, 2 (the remaining baseline work),
9 as findings move, then 10.

| # | Planned item | Where it is decided | Status | What remains |
| ---: | --- | --- | --- | --- |
| 1 | Engine: paged KV cache, continuous batching with preemption, prefix cache, OpenAI-style server | D-002 to D-005 | **Done**, measured on one RTX 5090 | Stretch goals: a fused paged-attention kernel, chunked prefill |
| 2 | Baselines: Hugging Face static batching and vLLM | D-006, D-015 | **Done** (vLLM at defaults) | A stronger baseline than static batching; vLLM matching-features run (the eager control is in pass 5); a second GPU class |
| 3 | Benchmark harness and honest results: load generator, write-once results, determinism self-test, machine monitor | D-014 to D-016, D-019 | **Done** | Keep it current |
| 4 | Speed work, only after profiling | D-003, D-020, D-021 | **Current.** Profiled; D-020 and D-021 pre-registered; pass 5 (capacity, attribution, eager control) running | Analyse pass 5; then run D-020 or fix the server layer, per the pre-written criteria |
| 5 | Router with prefix-cache-aware policies, validated on **real replicas** (the Phase 2 gate) | D-007, D-011, **D-022** | Built; compared only on a mock fleet. **Pre-registered (D-022); harness not yet changed; nothing measured** | Add a real-backend option to `bench/router_compare.py`, then run D-022 |
| 6 | One extension: a calibrated serving simulator (chosen) or speculative decoding (declined) | D-011, **D-024** | **Pre-registered (D-024); not started** | Measure step and prefill times, fit the timing model, validate on held-out cells |
| 7 | Deployment: Dockerfiles and Kubernetes manifests | STATUS | Not written | Write them; mark untested if Docker or a cluster is unavailable (Docker is not installed on the laptop) |
| 8 | What paging buys (raised by the independent review) | review, **D-023** | **Pre-registered (D-023); no run yet** | Run the block-size experiment at a binding memory budget |
| 9 | Review findings | `docs/independent-review-2026-11-24.md` | Tracked there | Follow its status table |
| 10 | Cleanup of the shared benchmark machine | STATUS section 6, item 13 | Pending | Delete `~/pagedserve_work` after the last GPU work, and confirm the home directory is back to baseline |
