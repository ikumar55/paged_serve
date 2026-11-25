# Second independent review (2026-11-24, evening)

A fresh chat was given only this repository and a neutral prompt that asked for a sceptical
assessment (resume value, whether the numbers hold, fairness, honesty, weaknesses, next steps)
without being told our conclusions. Its answer is reproduced **unedited** below the tracker. The
tracker, the verification notes and the plan are ours and are updated as findings move; the
response itself is never edited. The first review is `independent-review-2026-11-24.md`.

## Verification notes (ours, 2026-11-24 evening)

- **S2-1 (vLLM at 32 sequences against the engine at 64): verified, and the headline was fixed.**
  `results/vllm-rtx5090-ceiling` has vLLM at 64 sequences delivering 82.57 and 82.63 req/s at 96
  and 128 offered (both flagged "generator saturated", so a lower bound, one seed). D-019 already
  recorded this; the README, STATUS, roadmap and D-020 text quoted only the 32-sequence figure.
  All four places now say the gap is about 2.1 times with vLLM at 32 sequences and at least about
  2.85 times with vLLM at 64 sequences (the engine's own setting).
- **S2-3 (the gather arm lost requests to ConnectError): verified.** In `results/d025-sat-gather-*`
  the load generator logged a `ConnectError` for 170, 7, 153 and 82 of 1,920 requests (seeds 0 to 3)
  and the kernel and graph arms lost none. The D-025 text that said the cause "was not examined" is
  corrected. Why the server refused those connections under the gather path's overload is not yet
  investigated (candidates: the listen backlog, an event loop starved by the engine thread).
- Everything else below is the reviewer's claim, **not yet verified by us**, except where the
  tracker says otherwise. The reviewer did not run the test suite and did not read several files
  (`bench/baselines/hf_server.py`, `engine/prefix_cache.py`, `engine/paged_cache.py`, `router/`).

## Tracker

| # | Finding | Status | Where it is handled |
| --- | --- | --- | --- |
| S2-1 | vLLM compared at 32 sequences, engine at 64; at 64 vLLM gives 82.6 (lower bound), so the gap is at least 2.85 times | **Claim fixed** in README, STATUS, roadmap, D-020; the measurement itself is open | Roadmap item 11 (matched same-session comparison, D-030) |
| S2-2 | "2.1x" is against a deliberately naive gather path that pads to the longest sequence; a length-bucketed gather would be a fairer own-baseline | Open | Item 11 or 12: add a length-bucketed gather arm to the comparison (decide when planning D-030) |
| S2-3 | Gather arm lost requests to ConnectError | **Verified; cause found (2026-11-24):** in-flight requests peak at exactly 997 in all four gather runs and the first failure happens there; no other arm reaches it; consistent with a 1,024 open-file limit (not confirmed on the machine). Effect quantified: the kernel's gain over gather is 1.43 (1.33 to 1.54) in a steady-state window against 1.55 whole-run. Tool: `bench/inflight.py` | Item 12 (done); item 11 must record and raise `ulimit -n` for every system |
| S2-4 | The cross-session composites ("2.1x", "4.2 to 2.1") mix sessions | Labelled as rough in the docs; to be replaced by a same-session measurement | Item 11 |
| S2-5 | The Python load generator may cap vLLM (one run flagged generator-saturated) | Open | Item 11: use a load generator that is not the bottleneck (several processes or a faster client) |
| S2-6 | The graph arm is only mildly saturated (about 5% below the 32 req/s offered), so D-028 has no headroom | Open, important for D-028 | Offer at least 48 req/s to the graph arm when D-028 runs (amend the D-028 plan **before** any result exists) |
| S2-7 | D-026's metric has low power and uses the engine's own fp32 path as truth | **Planned 2026-11-24 as D-033** (KL and perplexity on real text, Hugging Face fp32 as the independent truth and Hugging Face bf16 as the yardstick); not built or run | Item 12 |
| S2-8 | D-027 is the most suspect result: replay unproven, and it sits uneasily with D-020's non-identical bf16 outputs | Open (rerun already planned) | Item 4 (the D-027 rerun with `graph_statistics`) |
| S2-9 | Framing: "diagnostic" sits next to a 2.1x headline in the first README paragraph | **Fixed 2026-11-24**: the README now opens with one table and three caveats (labelled diagnostics and failed gates, the rough vLLM comparison, one GPU and the in-flight ceiling) | Item 12 (done) |
| S2-10 | The tail of the work is not on GitHub and CI has not run on GPU-era commits | **Fixed 2026-11-25 (and the second half was wrong):** the work up to `c7e416c` was pushed with the owner's say-so, and CI turned out to have run green already on `a5a5ea3` (2026-11-24) and `ad0215a`, and runs green on `c7e416c` (lint, type check, CPU tests on Ubuntu). Later commits stay local until the owner says push | Item 12 (done) |
| S2-11 | Results record `git.commit: null` (a `git archive` was shipped) | **Fixed 2026-11-24** for new results: the environment capture falls back to `PAGEDSERVE_COMMIT` (Dockerfile build argument) or a `COMMIT` file (written by the GPU session scripts) and records the source; old results keep `null` | Item 12 (done) |
| S2-12 | Kernel: no isolated evidence (GB/s against the card's peak), no comparison with FlashInfer, flash-attn or vLLM's kernel; simple form (no split-KV, fixed `num_warps`) | Open | Item 13 (D-031) |
| S2-13 | The headline feature (paging) has no experiment of its own (D-023 unrun) | Open | Item 8 (D-023) |
| S2-14 | Router only on a mock fleet; D-022 and D-024 unrun; keep router numbers off the resume | Open | Items 5 and 6 |
| S2-15 | Generality: one GPU class, one 1B model, random-token prompts with `ignore_eos`, one run per cell; the kernel is shape-locked to GQA 4:1, head size 64, block size 16 | Open (stretch) | Item 14 (D-032) |
| S2-16 | The decision log (about 2,000 lines) is hard to skim | **Fixed 2026-11-24**: a summary table of all 28 entries is at the top of `decisions.md` | Item 12 (done) |
| S2-17 | Unexplained remainder to vLLM beyond graphs: unfused QKV and gate/up projections, one-at-a-time prefills, the Python step | Open | D-028 (attribution), then D-029 (batched prefill) |
| S2-18 | Mutation-testing claims were done by hand and cannot be rerun | Open (minor) | Item 12 if cheap: keep the mutation scripts in `tools/` |
| ours-1 | Found by us on a `kind` cluster, not by the reviewer: the router's `/health` stays 200 with no reachable replica (replicas are healthy until a request fails; the poll does not mark them down), so its Kubernetes readiness probe proves nothing | **Fixed 2026-11-24** (six tests) and re-checked on the `kind` cluster (router not Ready with no backend, Ready once engines are up, no restarts); recorded in `deploy/README.md` | Item 12 (done) |

## The response (unedited)

Verdict. This is an unusually well-evidenced project. Every headline number I recomputed from results/ reproduces, and the failures, retractions and post-hoc labels are in the record. It is a strong 7/10 resume project. The weak spots are the framing, not the arithmetic: the gap to vLLM is quoted with a handicapped vLLM, and the "2.1x" is measured against a deliberately naive gather path. The most suspect result (D-027) and the D-028 measurement are still open.

I did not run the test suite and could not check the 441-test claim. I did not read hf_server.py, prefix_cache.py, paged_cache.py or router/, and I cannot verify the vLLM version or the pre-registration timestamps.

### 1. Resume value: 7/10

Impressive

- Method. Write-once results, a determinism self-test that caught a faulty GPU (D-016), a machine monitor and audit, pre-registered predictions that were recorded as wrong (graphs predicted 1.10–1.20x, measured 1.35–1.44x), and two failed gates left failed.
- Profile-driven optimisation. It found that the step is GPU-bound on the real length mix (D-020 prerequisite), wrote a Triton kernel, then CUDA graphs. Its documented profile of the real-mix step at batch 60 (the profile_decode tool) shows a 52% shorter step with the kernel and 16% more with graphs.
- Code quality. It is clean and strictly typed. The kernel (triton_paged_attention.py:30-103) and the graph decoder, with its scratch block for padding rows (cuda_graph.py), are readable and sound.

Merely competent

- The paged cache, block allocator, prefix cache and scheduler are a faithful vLLM reproduction on a 1B model, on one consumer GPU.
- The kernel is the simple form: one program per (sequence, KV head), no split-KV, fixed num_warps=2 (line 155), shape-locked to GQA 4:1, head size 64, block size 16.
- It is still about 2–3x behind vLLM.
- The baseline is HF static batching, which is weak by construction.
- The router has only run against a mock backend.
- The 1,951-line decision log shows rigour but is hard to skim.

What would move it: 8–8.5 with next steps 1–3 below.

### 2. Do the numbers hold? Yes, with small exceptions

I recomputed delivered requests per second as completed requests divided by the time from first send to last completion, from each run's .jsonl.

| Arm (batch 64, 32 req/s offered) | Per seed | Mean | Docs |
| --- | --- | --- | --- |
| Gather (d025-sat-gather) | 13.38, 15.17, 13.05, 14.30 | 13.97 | 13.97 ✓ |
| Kernel (d025-sat-kernel) | 21.51, 21.56, 21.95, 21.35 | 21.59 | 21.59 ✓ |
| Kernel, same-session control (d020-sat-eager) | 20.21, 20.96, 21.02, 20.77 | 20.74 | 20.74 ✓ |
| Kernel plus graphs (d020-sat-graphs) | 29.05, 29.72, 28.46, 28.67 | 28.98 | 28.98 ✓ |

Paired ratios. Kernel over gather is 1.61, 1.42, 1.68, 1.49, mean 1.55 ✓. Graphs over eager kernel is 1.44, 1.42, 1.35, 1.38, mean 1.40 ✓. The cross-session composite (graphs over gather) is 2.07 directly, or 2.08 paired by seed ✓.

vLLM. vLLM at 32 sequences gives 59.63 req/s, and the vLLM-to-engine ratio is 4.51, 3.71, 4.28, 4.39, mean 4.22 ✓. Graphs against that vLLM figure is 2.06 ✓.

Steady-state check. Whole-run throughput includes ramp-up and drain. Counting completions in the window 15–55 s gives gather 14.8, kernel 21.6, eager 21.4 and graphs 30.3. The ratios barely move (1.46 and 1.41).

Graphs arm. It is only mildly saturated: 115–233 requests were in the system at t=55 s, and the arm sits within about 5% of the 32 req/s offered. D-028 would have no headroom to show a further gain at this rate.

### 3. Fairness

- vLLM config. vLLM runs with 32 sequences against the engine's 64 (bench/sweep.py:188). The same vLLM at 64 sequences delivered 82.6 req/s (vllm-rtx5090-ceiling/b64-w0-r128, one seed, flagged generator-saturated, so a lower bound). Against that, the gap to the graphs engine is at least 2.85x, not 2.1x. The earlier review raised this as open item S2b, and the headline still uses the lower vLLM figure.
- Sessions. vLLM (pass 5) and the engine arms were measured in different sessions. The kernel arms agree across sessions to 4%, which helps, but the "2.1x" and "4.2x → 2.1x" are cross-session composites. They are labelled as rough in the docs.
- Load generator. vLLM's 59.6 is 93% of the 64 req/s offered, and one run is flagged generator-saturated. The Python load generator may be capping vLLM.
- Baseline pathology. The gather path pads every sequence to the longest in the batch (3.7x extra positions). The 1.55x is largely "removing a padding pathology" in this design, on a workload whose lognormal length mix makes that pathology large. A length-bucketed gather would be a fairer own-baseline.
- Dropped connections. The gather arm lost 170, 7, 153 and 82 of 1,920 requests to ConnectError, and the kernel and graph arms lost none. D-025 says "the cause was not examined" (decisions.md:1656), but the summary lines name it. These runs are in a different regime, with connection drops. The effect on the ratio is unquantified and probably small.
- Scope. One GPU class, one model, random-token prompts with ignore_eos, four workload seeds with one run per cell. The kernel only supports the Llama-1B head layout.

### 4. Honesty: good, with three soft spots

What is done well. Failed gates stay failed. The post-hoc D-026 and D-027 criteria are labelled and written before their runs. Wrong predictions are recorded. The README and STATUS carry the caveats.

- D-026's metric has low power. The mean maximum logit error of about 0.15 is dominated by bf16 rounding of logits of size 16–32, which are spaced 0.0625–0.125 apart (decisions.md:1760-1764). Small attention-rounding differences would be invisible to it. The "truth" is also the engine's own fp32 path.
- D-027 is the most suspect result. The graph and eager-kernel errors are identical at all 12,600 positions, but graph_statistics is not in the result files, so replay is unproven (decisions.md:1832-1839). It sits uneasily beside D-020's finding that the bf16 outputs are not hash-identical in any of six configurations. The rerun has not been done.
- Framing. "Diagnostic, not a claim" sits next to a 2.1x headline in the first README paragraph. The caveats are there, but a skimmer takes away "2.1x".
- Not published. HEAD is 775659d, but the branch was pushed only to ad0215a (STATUS says so). The tail of the work is not on GitHub, and CI has not run on any GPU-era commit.

### 5. Biggest weaknesses

- bf16 is unresolved. The kernel flipped one near-tie (0.1875 against the 0.125 limit) and the cause was never found. fp32 is solid; the bf16 evidence is aggregate only.
- The kernel has no isolated evidence. There is no bandwidth or roofline figure and no comparison with FlashInfer, flash_attn_with_kvcache or vLLM's own kernel.
- Unexplained remainder. Most of what separates the engine from vLLM beyond graphs is untested: unfused QKV and gate/up projections (model.py:139-173), one-at-a-time prefills (scheduler.py:302-329), and the Python step. D-028 would have measured it; it is built but unrun.
- The headline feature has no experiment. D-023 ("what paging buys") is unrun. The default path copies blocks into a padded tensor, so only the opt-in kernel actually reads pages in place.
- Reproducibility. Results record git.commit: null (a git archive was shipped to the box), and code hashes exist only for pass 5. The environment is a private shared fleet with no Docker or one-command reproduction. The mutation-testing claims were done by hand and cannot be rerun.
- Generality. A 1B model is launch-bound and attention-padding-heavy in a way larger models are not, so the speedups may not transfer.
- Router. It is validated only on a simulated fleet, and the roadmap's D-022 to D-024 are not run. Keep router hit-rate numbers off the resume.

### 6. Next steps, ranked

1. Matched vLLM comparison, same session (0.5–1 GPU day). Run vLLM at 32 and 64 sequences, offer at least 1.5x capacity to every system (at least 48 req/s for the engine), and use a load generator that is not the bottleneck. Interleave by seed in one session. This gives a defensible "engine at X% of vLLM".
2. Credibility sweep (about 0.5 day, mostly CPU).
   - Rerun D-027 with graph statistics and run D-028.
   - Find the cause of the gather arm's ConnectErrors.
   - Record commit hashes in results.
   - Push and get CI green.
   - Add a sensitive bf16 accuracy test: KL or perplexity on real text against fp32 and HF bf16.
   - Cut the README to a short summary with one table and the three caveats.
3. Kernel microbenchmark (1–2 days). Compare against FlashInfer, flash-attn and vLLM's kernel across batch and context length. Report GB/s against the card's peak. Try split-KV and autotuning. This is what ML-systems interviewers will probe.
4. D-023, what paging buys (about 1 day). Measure concurrency at a fixed memory budget against contiguous preallocation, with a block-size sweep. It gives the headline feature its own experiment.
5. Generality (1–2 days). Repeat the kernel-plus-graphs diagnostic on one more model (3B or 8B) or GPU class, with real text and EOS-terminated lengths instead of random tokens.
