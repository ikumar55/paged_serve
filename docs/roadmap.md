# Roadmap: planned scope against what is done

The original project plan (a gitignored `PROJECT_PLAN*.md`) was lost in a laptop reset and could
not be recovered from the repository, its history, the owner's OneDrive or the earlier session
transcripts. This roadmap is **reconstructed** from `docs/STATUS.md`, `docs/decisions.md` (D-011
refers to a "Phase 2 gate" at the end of Week 4 and a six-week plan) and the owner's local notes.
Where the original plan had items this list does not, they are missing; add them. A living
document: update the status column as work lands.

Last updated: 2026-11-24.

| # | Planned item | Where it is decided | Status | What remains |
| ---: | --- | --- | --- | --- |
| 1 | Engine: paged KV cache, continuous batching with preemption, prefix cache, OpenAI-style server | D-002 to D-005 | **Done**, measured on one RTX 5090 | Stretch goals: a fused paged-attention kernel, chunked prefill |
| 2 | Baselines: Hugging Face static batching and vLLM | D-006, D-015 | **Done** (vLLM at defaults) | A stronger baseline than static batching; vLLM matching-features and eager runs (eager is in pass 5); a second GPU class |
| 3 | Benchmark harness and honest results: load generator, write-once results, determinism self-test, machine monitor | D-014 to D-016, D-019 | **Done** | Keep it current |
| 4 | Speed work, only after profiling | D-003, D-020, D-021 | Profiled; two experiments pre-registered, **nothing implemented** | Attribute the server's loss (D-021, pass 5), then run D-020 or fix the server layer, per the pre-written criteria |
| 5 | Router with prefix-cache-aware policies, validated on **real replicas** (the Phase 2 gate) | D-007, D-011 | Built; compared only on a **mock fleet** with placeholder timing | Run 2 to 3 real engine replicas (on one GPU if need be, stated as a confound) behind the router with a shared-prefix workload, round-robin against the prefix-aware policies |
| 6 | One extension: a calibrated serving simulator, or a speculative-decoding study | D-011 | **Not started**; decision open | Recommended: the calibrated simulator (the real step-time and prefill data now exist to calibrate the mock); speculative decoding is the riskier alternative |
| 7 | Deployment: Dockerfiles and Kubernetes manifests | STATUS | Not written | Write them; mark untested if Docker or a cluster is unavailable |
| 8 | What paging buys (raised by the independent review) | `docs/independent-review-2026-11-24.md` | No experiment | Concurrent sequences or memory waste at a fixed memory budget, paged against a contiguous cache |
| 9 | Review findings | `docs/independent-review-2026-11-24.md` | Tracked there | Follow its status table |
| 10 | Cleanup of the shared benchmark machine | STATUS section 6, item 13 | Pending | Delete `~/pagedserve_work` after the last GPU work, and confirm the home directory is back to baseline |

Suggested order: finish pass 5 and D-021; one real speed result (item 4); the router on real
replicas (item 5); the extension (item 6); deployment files and the paging experiment (items 7 and
8); then cleanup (item 10).
