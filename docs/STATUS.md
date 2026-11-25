# Project status and handover

A running record of what has been built, how it was checked, what is still open, and how to pick
the work up. **Update this file at the end of every step.** It complements
[`decisions.md`](decisions.md), which holds each design decision with its measured effect; where
the two disagree, `decisions.md` is the record and this file is the map.

Last updated: 2026-11-24 (evening), after the speed work: the in-place decode-attention kernel
(D-025), its accuracy standard (D-026), CUDA graphs on top of it (D-020 re-test) and their accuracy
standard (D-027); D-028 (where the engine thread's time goes now) is planned and built on the CPU
but not measured. Everything before that is the 2026-11-23 benchmark on a Capsule RTX 5090 machine
(D-019: engine, Hugging Face static batching and vLLM, four workload seeds each, determinism
self-test before every system; shared-prefix runs, a machine monitor and audit). The numbers are
in section 3, D-019 and the D-020 to D-028 entries. **Read the speed claims with the caveats:** the
fixed bf16 gates of D-025 and D-020 were missed (recorded as failed), the later accuracy standards
were chosen after the fact (labelled), and the speed figures are labelled diagnostics. **A machine
in Oxmiq's Capsule fleet still holds project files (`~/pagedserve_work`, kept because more runs
are plausible); the cleanup duty in section 6, item 13 applies. The branch is pushed up to
`ad0215a`; later commits are local only.**

## 1. The project

**pagedserve** is a small LLM inference engine written from scratch in Python and PyTorch: a paged
KV cache, continuous batching, prefix caching, and a prefix-cache-aware router for several
replicas. The model is Llama-3.2-1B-Instruct (D-001). It is compared, on one GPU model, against
Hugging Face `generate` with static batching and against vLLM (D-006). Ideas come from public
papers (attributed in `decisions.md`); no code is copied.

The original project plan (a `PROJECT_PLAN*.md`, gitignored) is not in the repository and was not
recovered after a laptop reset. The reconstruction below is from `decisions.md` and the commit history; D-011 says one extension
(a serving simulator, or a speculative-decoding study) is chosen after the router gate.

## 2. Component status

"Verified" means checked by the tests described, on the hardware named. Since 2026-11-23 the engine
server (and so the scheduler and paged cache) has run on one GPU, a Capsule RTX 5090 (D-019), for a
20-second smoke run and a greedy evaluation; the router has still only run on the mock backend.

| Component | Code | Verified how | Not verified |
| --- | --- | --- | --- |
| Llama forward pass, contiguous KV cache (D-002, D-012) | `engine/model.py`, `config.py`, `weights.py`, `kv_cache.py`, `generate.py` | Bit-identical logits to Hugging Face in fp32 on CPU (50 prompts, a 2048-token prompt); within the 1e-3 target on an RTX A6000; greedy decoding identical on 200 of 200 prompts in fp32 and bf16 on the A6000 and in bf16 on the RTX 5060 Ti. **RTX 5090:** fp32 logits bit-identical to Hugging Face on the same GPU (0.0) but 1.19e-3 from Hugging Face on the CPU, which is Hugging Face's own GPU-versus-CPU gap on that card, so the old target of 1e-3 against the CPU failed there; the GPU contract was changed on 2026-11-24 to equality with Hugging Face on the same GPU (1e-4) and no further from the CPU than Hugging Face's own GPU run (D-019) | Other GPUs; kernels other than plain PyTorch SDPA |
| Block allocator, paged KV cache (D-003) | `engine/block_allocator.py`, `paged_cache.py` | Paged logits exactly equal the contiguous cache's, with scattered blocks; deliberately broken variants fail | GPU; the gather-based attention path has never been profiled |
| Batched decode and scheduler (D-004) | `engine/scheduler.py`, `paged_cache.py` (`PagedBatchCache`) | Equals single-request greedy decoding on a tiny model, and equals Hugging Face `generate` on the **real weights**, fp32, CPU, 16 prompts x 24 tokens, including 9 preemptions by recompute (`tests/test_real_llama_scheduler.py`) | bf16, GPU, the 200-prompt evaluation through the scheduler, any speed |
| Prefix cache (D-005) | `engine/prefix_cache.py`, `block_allocator.py` | Tiny-model tests incl. eviction and preemption; six broken variants fail | Real weights, bf16, hit rates on real traffic; reused blocks may differ from an uncached run in rounding |
| Engine server | `engine/server.py` | Over real sockets on a tiny model: output equals reference, streaming, EOS, disconnects abort and free blocks, the load generator drives it. **On an RTX 5090 (2026-11-23):** served the real weights in bf16 (batch 32, 8 GB KV cache), 40 of 40 requests of a 20 s, 2 req/s smoke run succeeded (TTFT p50 32 ms, inter-token p50 9.3 ms; a smoke run, not a benchmark); `bench/greedy_eval.py` bf16 50 of 50 prompts identical to Hugging Face, teacher-forced top-1 3200 of 3200, max logit difference 0.0 (`results/greedy-eval-rtx5090-bf16.json` on the box, see D-019) | The scheduler path against Hugging Face on a GPU (the evaluation tests the plain engine); `--kv-cache-gb` sizing beyond 8 GB; throughput (sweeps pending) |
| Mock backend (D-008) | `engine/mock.py`, `server.py --backend mock` | Output is a pure function of the last token and position, so batching/preemption/routing cannot change it; pool accounting matches a real model | **Timing is uncalibrated placeholders**; only the decode step is loosely based on D-015 |
| Router (D-007) | `router/policies.py`, `app.py`, `server.py` | Mock replicas and router over real sockets: pass-through, the four policies, load cap, failover, 503, disconnect propagation, load generator through the router; seven of eight broken variants fail (the eighth is an equivalent mutant); `--max-entries` (the approximate policy's memory limit) reaches the policy from the command line | Real engine replicas, GPU, mid-stream replica death. The mock-fleet comparison below is not GPU evidence |
| Router on real replicas (D-022) | `bench/router_compare.py --backend llama`, `deploy/gpu_d022.sh` | **CPU only (2026-11-25):** the option starts real `engine.server` replicas, warms each up directly, labels the result as real replicas sharing one GPU and keeps the weights path out of the file; 14 tests, 12 deliberate breakages caught; the mock path and its 20 tests unchanged | **Never run with real replicas**; no result; waits for the machine (about one hour) |
| Router policy comparison (D-007) | `bench/router_compare.py` | Replicas and router as real subprocesses over real sockets, mock backend; 37 deliberately broken variants of the script, the router flag and the policy factory (including the per-variant load cap) each make a test fail. Writing them found one real flaw (a refused write could leave half a result behind, now fixed) and a test that the digest of the workload was recorded from what was meant rather than what was sent (now from what is sent); results in section 3 | Anything about a GPU; Linux; more than 3 seeds and one workload shape; the load cap's effect beyond the two settings tried |
| Load generator and workloads (D-014) | `bench/loadgen.py`, `workload.py`, `client.py`, `runner.py`, `metrics.py`, `results_io.py`, `plots.py` | Calibrated against a fake server; pinned hashes prove old workloads still regenerate exactly; optional shared-prefix mode. **Since 2026-11-25 (D-030):** `--shards K` runs the same requests from K processes, and every summary has the event-loop lag, the in-flight peak, a steady-state-window completion rate and the open-file limit | The sharded path is tested on Windows and on Linux in Docker against a server that answers at once; **never used against the engine or vLLM on the GPU machine** |
| Matched vLLM comparison tooling (D-030) | `bench/sharded.py`, `bench/null_server.py`, `bench/fdlimits.py`, `bench/matched_report.py`, `deploy/gpu_d030_lib.sh`, `gpu_d030_a.sh`, `gpu_d030_b.sh`; the sweep manifest records both open-file limits | **CPU only (2026-11-25):** the report applies the validity rules, the probe-to-rate rule, the paired ratios, the wording rule and the six predictions of D-030 to synthetic sweep directories (39 tests); the sharded generator and the null server are tested on real sockets and processes (15 tests), the new metrics and limits in 19 more, the sweep's limit fields in 3; 37 deliberate breakages, all caught; also run on Linux in Docker against the null server | **Never run on the GPU machine**; no figure; the optional length-bucketed gather arm is not built |
| HF static-batching baseline (D-015) | `bench/baselines/`, `bench/sweep*.py` | Swept on an RTX A6000 (D-015) and again on a Capsule RTX 5090, four seeds, rates 4/8/16 at batch 32/64 (section 3, D-019) | Only 4 workload seeds; its best setting (batch 64) is still at the edge of the grid |
| Sweep tooling for engine and vLLM | `bench/sweep.py` | Command construction, validation and manifests tested with fake servers; the hf-static commands are pinned to what the recorded sweeps used | Ran real engine, Hugging Face and vLLM 0.34.0 sweeps on the RTX 5090 (D-019), so the vLLM flags worked there; the box's code was a `git archive` of b101006, so env captures record `git.commit: null` |
| GPU determinism self-test (D-016) | `bench/gpu_selftest.py` | Caught a faulty GPU once (results from it were retracted) | - |
| GPU-box setup | `deploy/setup_gpu_box.sh`, `docs/gpu-runbook.md` | Ran on a Capsule RTX 5090 machine on 2026-11-23 (with `ALLOW_OLDER_DRIVER=1`, because the fleet's drivers advertise CUDA 13.0): built the engine environment and the self-test passed. The vLLM environment step and the sweep commands also ran (`deploy/gpu_pass1.sh`, `deploy/gpu_pass2.sh`; D-019) | A rented provider |
| Decode attention that reads blocks in place (D-025, opt-in `--paged-attention-kernel`) | `engine/paged_attention.py` (float64 reference, CPU-runnable), `engine/triton_paged_attention.py` (Triton kernel), plumbing in `paged_cache.py`, `model.py`, `scheduler.py`, `server.py`, `bench/scheduler_eval.py`, `bench/profile_decode.py` | **CPU only (2026-11-24):** the reference agrees with the gather path in float64; the in-place route gives the same logits and scheduler tokens (preemption and prefix-cache hits included) as the gather path; the function is really called; breakages caught (one equivalent mutant) | **GPU (2026-11-24, RTX 5090, D-025):** compiles; error against float64 at most 0.57 times the gather path's in fp32 and equal in bf16; 15 of 15 GPU tests; scheduler 50 of 50 identical to Hugging Face in fp32 (frozen and shared-prefix, preemptions and cache hits). **Gate (a)(ii) FAILED in bf16:** one divergence (shared-prefix prompt 10, token 3) has a logit gap of 0.1875 against the fixed 0.125, and stayed after the one allowed fix (rounding the softmax weights to bf16, which moved it between configurations; reproducible, outputs bit-identical on repeat); the gather path gets that prompt right. Kernel reverted to its first version; nothing claimed (D-025). **Diagnostic speed run, not a claim:** real-mix step 52% shorter (38 to 60%), engine saturated throughput 1.55 times the gather path's (paired per seed: 1.61, 1.42, 1.68, 1.49), zero failed requests; worth a D-026 for the bf16 question. **D-026 (post-hoc criteria, written first):** against fp32, over 12,600 positions, the kernel's bf16 logit error is the gather path's (mean ratio 1.001, p99 0.999; clear flips 4 against 9), so the speed figures may be quoted with both facts stated (D-025's literal bf16 gate was missed by one near-tie). Opt-in; the gather path stays the default. **CUDA graphs on top (D-020, `--cuda-graphs`, `engine/cuda_graph.py`):** fp32 hash-identical to the eager kernel path and 50 of 50 against Hugging Face; bf16 not hash-identical, so the pre-registered strict gate is recorded as failed and nothing is claimed; a labelled diagnostic: step 16% shorter, saturated throughput 1.40 times the eager kernel path (1.35 to 1.44; predicted 1.10 to 1.20), about 2.1 times the gather path in total. **D-027 (post-hoc criteria, written first):** the graph path's bf16 logit error against fp32 equals the eager kernel's at all 12,600 positions and meets D-026's four criteria against the gather path; open: a rerun that records the replay counts (not yet done). **D-028 (planned, not measured):** the scheduler now times its phases (room, decode, admission) and counts prefills, reported in `/info`; `bench/phase_report.py` and `deploy/gpu_d028.sh` are ready, the measurement is waiting for the GPU machine |
| KL and perplexity accuracy test (D-033) | `bench/kl_perplexity.py`, `deploy/gpu_d033.sh` | **CPU only (2026-11-25):** 25 tests on a tiny model (a stand-in for Hugging Face, the float64 reference in place of the kernel; a broken attention path is detected end to end); 13 deliberate breakages caught; the fixed text gives 73,476 tokens (288 windows available) and the extended set 103,229 (404); `logits_to_keep` confirmed in the installed transformers | **Never run on the real model** (this laptop has 15.7 GB of memory, short of the ~15 GB the four model copies need); no accuracy result exists; waits for a machine session |
| Dockerfiles, Kubernetes manifests (roadmap item 7) | `deploy/docker/` (engine, router, bench Dockerfiles; a compose file for a no-GPU mock stack), `deploy/k8s/` (base: engine StatefulSet with a headless Service and one GPU per pod, router Deployment and Service, weights volume claim; overlays `kernel-graphs` and `mock`), `deploy/README.md`, `deploy/validate.{sh,ps1}`, `tests/test_deploy_files.py` | **Offline, on Docker and on a real local Kubernetes cluster (2026-11-24):** the three images build; the mock stack runs on Docker Desktop (WSL 2) and, through the `mock` overlay, on a `kind` cluster: all pods Ready on the first apply, the router finds both replicas by StatefulSet DNS name, 160 of 160 load-generator requests succeed, the pod security settings are enforced (uid 10001, read-only root), deleting an engine pod during traffic loses no request (45 of 45) and the StatefulSet recreates it, and the `single-gpu` overlay on a GPU-less cluster stays Pending with `Insufficient nvidia.com/gpu`; hadolint clean; the three kustomizations render and pass kubeconform against the Kubernetes 1.34 schemas; 24 tests parse every argument list with the servers' real parsers and check names, ports, probe routes, image tags and users across files, the overlays' effects, and that the router image needs no PyTorch; 21 deliberate breakages of the files, all caught; **found by running it, and fixed: the router's `/health` was optimistic (healthy until a request fails), so its readiness probe proved nothing at startup; it now asks the replicas until one has answered** (six tests, all deliberate breakages caught; re-run on the cluster: router not Ready with no backend, Ready within 15 s once the engines start; its liveness probe is now a TCP check so it is not restarted while the engines load) | **No GPU run, in a container or on a cluster:** GPU visibility in the container, Triton compiling there, the weights volume and scheduling two GPU replicas onto two GPUs are untested (`deploy/README.md` lists what to try first) |

Test suite (CPU), `pytest -m "not gpu and not real_model"`: **`619 passed, 3 skipped, 12
deselected` on Windows on 2026-11-25** (with the two timing tests below deselected by hand), about
200 s. The deselected are the GPU and
real-weights tests (run by hand on the GPU machine) and the two load-generator timing tests
`test_runner.py::test_every_request_completes_in_order_at_its_scheduled_time` and
`test_loadgen_cli.py::test_end_to_end_over_a_real_socket`, which fail intermittently on Windows
only (15.6 ms clock tick; section 6) and which I left out of this count by name; they passed on
Linux (`323 passed, 10 deselected` on the RTX 5090 machine on 2026-11-23, before about 120 tests
were added). The 3 skips are the two GPU test files that need Triton (not installed on the
laptop) and the machine monitor's `/proc` test (Linux only). The GPU tests of the kernel
(15) and of CUDA graphs (4) passed on the RTX 5090 machine on 2026-11-24. The suite has not been run
on Linux since the tests added on 2026-11-24 except by CI: GitHub Actions (Ubuntu, CPU torch: lint, type check and the CPU tests) ran green on the pushes of 2026-11-24 and 2026-11-25 (`a5a5ea3`, `ad0215a`, `c7e416c`).

## 3. Measurements so far

The RTX 5090 results are summarised after the A6000 paragraph below; D-019 has the tables.

All of this first paragraph is on an RTX A6000 (workstation GPU, not a datacenter one), Llama-3.2-1B-Instruct, bf16, one run
per cell unless stated; full method and caveats in D-015:

- **Tuned Hugging Face static-batching baseline:** max batch size 32, batch window 20 ms (batch 64
  is equivalent, 100 ms window indistinguishable): about **5.3 req/s (760 tokens/s) at
  saturation**, with 57% of decode slots wasted in the one run that measured it.
- **Run-to-run spread is 10 to 19%** of the mean (four seeds per cell in `results/variation.md`), so
  differences of that size mean nothing.
- Batch 16 saturates near 4.0 req/s (579 tokens/s); a batch window of 0 turns batching off.

**RTX 5090 (2026-11-23, D-019): the three-way comparison**, same model, bf16, standard workload,
mean of four workload seeds (min to max), delivered throughput in req/s at 16 req/s offered:
Hugging Face static (batch 64, 20 ms window) **9.0 (8.4 to 9.4)**, this engine (batch 64) **14.4
(13.5 to 15.1)**, vLLM **15.8 (15.5 to 16.2)**. **Read that cell with care:** the metric is
capped by the arrivals (about 15.95 req/s at 16 offered, for these seeds), so it barely tells the
engine and vLLM apart; capacity is the saturated rate. The engine keeps up to about 12 req/s and
is at its edge at 16; Hugging Face queues from 4 req/s and saturates near 7 to 9 req/s.
**Capacity (pass 5, offered above capacity, four seeds paired, `results/saturation-rtx5090.md`):**
engine batch 64 **14.2 req/s (13.2 to 16.0)**, 2,050 tokens/s; vLLM (32 sequences) **59.6 req/s
(58.7 to 60.8)**, 8,630 tokens/s, so **vLLM delivers 4.2 times the engine (3.7 to 4.5)**;
Hugging Face's best is 9.0 req/s, so the engine leads it by about 1.6 times. With CUDA graphs
disabled vLLM falls to about 29 req/s, so graphs are worth about 2 times of its capacity and the
engine is still about 2 times behind eager vLLM. The engine's inter-token latency is 9 to 11 ms at
low load against vLLM's 2 to 3 ms, and 22 to 27 ms at 16 req/s; the decode-step profile (D-003)
shows the step is launch-bound at small batches, and at saturation the engine thread is busy
inside the scheduler step 97 to 98% of the time, a step about 2.5 times slower than the
fixed-length profile. The follow-up (D-021) explains most of that: offline, on the server's own
requests, the step costs 0.38 ms per token, twice the fixed-length figure, so the real length mix
is the main cause; the prefix cache costs under 1%; the server adds a further 19 to 31% that is
not explained. **Profiled on that mix, the saturated decode step is GPU-bound**: 14.6 to 25.4 ms,
the GPU busy 88 to 93% of it, attention plus the gather 84% of GPU time, and the padded batch
handles 3.7 times the positions it needs; so CUDA graphs can give at most 1.07 to 1.14 times
(D-020 is parked) and the next speed work is attention that reads the blocks in place (D-025). Spread across seeds is 2 to 16%. One GPU, one model, one
workload shape. Tables: `results/variation-*-rtx5090.md`.

**Speed work on the RTX 5090 (2026-11-24; labelled diagnostics, four seeds paired and
interleaved within one session, machine audit clean; caveats in D-025, D-020, D-026, D-027):**
the in-place Triton decode-attention kernel makes the real-mix decode step 52% shorter (20.1 to
9.4 ms) and saturated engine throughput 1.55 times the gather path's (13.97 to 21.59 req/s, paired
1.61, 1.42, 1.68, 1.49); CUDA graphs on top add 1.40 times (20.74 to 28.98 req/s, paired 1.44,
1.42, 1.35, 1.38) and cut the inter-token latency p50 from about 16 to about 10 ms. In total about 2.1
times the gather path. Against vLLM (an earlier session, not remeasured): 59.6 req/s with 32
sequences is about 2.1 times this engine (from 4.2 times), but with 64 sequences, the engine's own
setting, vLLM delivered at least 82.6 req/s (a lower bound, load generator saturated, one seed),
so the like-for-like gap is at least about 2.85 times. A same-session matched comparison is planned
(roadmap item 11). Accuracy: fp32 output is identical to Hugging Face, and against fp32 the bf16
error of the kernel and of the graph path equals the gather path's (mean logit error 0.1497
against 0.1495; D-026, D-027, criteria chosen after the fact); the strict bf16 gates fixed in
advance were missed by a near-tie and are recorded as failed.

**Prefix cache (shared-prefix workload, 8 x 1024-token prefixes):** the engine's hit rate was 89 to
92% with the cache on; median time to first token fell 17 to 22% (32 against 41 ms at 4 req/s) and,
where the engine is overloaded (16 req/s), throughput rose 11% (12.2 against 10.9). vLLM's median
time to first token fell 24 to 39%. The absolute saving is only 8 to 14 ms (short prefill on this GPU).

**Was the machine consistent?** A monitor now records other users' CPU, foreign GPU processes and
GPU throttling (`bench/machine_monitor.py`, `bench/machine_audit.py`). Over 55 minutes other users'
CPU peaked at 0.015 cores and nothing else touched the GPU; the 29 runs it covered audit clean. The
earlier runs were not monitored, so part of pass 2 was repeated under it: 17 pairs agree to 1.3% on
average (max 4.6%). The sweep's own contention flag is weak (six one-sample blips, cause unknown).
`results/machine-audit-p3-p4.md`, `results/repro-vs-original-rtx5090.md`.

D-017 chose an RTX 5060 Ti (16 GB) machine, validated the stack on it (greedy evaluation: 200 of
200 identical, bf16) and never benchmarked there. Do not quote the A6000 numbers next to anything
measured on another GPU, or the 5090 numbers next to either.

D-016: one RTX PRO 6000 Blackwell returned different results for identical inputs; results from
it are retracted. Every GPU measurement must be preceded by the self-test.

### Mock-fleet router comparison (not GPU, timing uncalibrated)

Five result files from `bench/router_compare.py` (4 mock replicas and the router as processes on
this Windows laptop, 3 seeds each, 600 or 1200 requests per run; details, counters and caveats in
D-007, files
`results/router-compare-mock-*.json` and `.md`, every one labelled mock/uncalibrated). Do not quote
these as performance. What they do show, about routing and plumbing:

- **The load cap decides whether prefix routing happens at all.** In the two overloaded runs (A, B)
  the default cap of 16 sent 85 to 90% of the prefix policies' requests elsewhere, so all four
  policies looked alike. In a later sweep at a load that was *not* overloaded (29 of 30 req/s, no
  failures, 1200 requests) the cap mattered sharply between 8 and 16: hit rate 44% at cap 4, 52 to
  55% at 8, then 70 to 72% at 16, 32 and off, identical from 16 up. So 16 is fine at that load and
  too low when replicas hold more requests in flight than that.
- **With the cap off and a lighter load (C), prefix-aware routing clearly raised the fleet hit
  rate:** approximate 70.9%, precise 67.8%, against round-robin 42.2% and least-outstanding 43.7%
  (three seeds, ranges do not overlap). TTFT p50 fell (78 to 64 ms in a placeholder-timed mock), p99
  did not, and work concentrated on the replicas holding the hot prefixes (computed-token imbalance
  1.3 against 1.06).
- **The approximate policy's memory matters a lot:** limited to 1024 chunks 64.2%, to 256 chunks
  46.2% (about least-outstanding). It did **not** lag the precise policy; it was higher by about 2
  to 3 points. Polling the precise policy 10 times faster narrowed that to 0.4 points, but the
  ranges overlap, so the poll-staleness explanation is **not established** (its no-match count
  barely moved). Other explanations are untested (D-007).
- **No cap value gave both a high hit rate and an even spread:** cap 8 was balanced (imbalance
  1.07) with 52% hits, cap 16 and up gave about 71% with imbalance 1.2 to 1.4.

The load cap therefore has to be chosen against the number of requests in flight (open item), and
the comparison has to be repeated on a calibrated mock and on real replicas before any of this is
called a router result.

## 4. How the work is done (conventions to keep)

- **Correctness before speed.** Every optimization lands with a test showing outputs did not
  change; the reference is Hugging Face (and, for the scheduler, `generate` on the real weights).
- **Honest claims.** Say what was measured, on what, once or repeatedly, and what was *not*
  measured. State where this engine is slower. Never present mock-backend output as GPU
  performance. If a result is later found wrong, retract it in `decisions.md` (D-002, D-016).
- **Prove the tests can fail.** For each new behavior, break the code in realistic ways and check
  a test fails. This caught real test gaps several times (e.g. a disconnect test that passed
  without the abort; a router workload that accidentally aligned with round-robin). Report
  survivors honestly, including equivalent mutants.
- **One step, one commit**, with a message that says what was checked and what was not. Do not push
  unless asked.
- Lint, types and tests must pass: `ruff check .`, `ruff format --check .`, `mypy` (strict), and
  `pytest -m "not gpu and not real_model"`.
- Result files are write-once and must not contain hostnames, IPs, usernames, GPU serials or
  instance ids; label results by hardware name (a machine id once leaked into a label; a leak scan
  caught it). Never commit weights, tokens or keys.
- Tooling note for agents on this Windows laptop: long here-documents in the Bash tool mangled
  backslashes and quotes more than once (broke a README example and a patch script). Use the file
  Write/Edit tools for anything containing backslashes, and PowerShell for running pytest.

## 5. How to run things

```bash
# CPU checks (Linux: `make check`; the Makefile expects .venv/bin/python, override with PY=python)
ruff check . && ruff format --check . && mypy && pytest -m "not gpu and not real_model"
pytest -m "real_model and not gpu" -s          # needs models/Llama-3.2-1B-Instruct/ (gitignored)
pytest -m gpu -s                               # needs a CUDA GPU; run by hand

python -m engine.server --model-dir models/Llama-3.2-1B-Instruct --dtype bfloat16 \
    --max-batch-size 32 --kv-cache-gb 8 --port 8000
python -m engine.server --backend mock --port 8001          # no model, no GPU
python -m router.server --replica http://127.0.0.1:8001 --replica http://127.0.0.1:8002 \
    --policy precise-prefix --port 9000
python -m bench.loadgen run --url ... --rate 4 --duration 60 \
    --shared-prefix-groups 8 --shared-prefix-len 512 --shared-prefix-skew 1 --out results/x.jsonl
python -m bench.sweep --system {hf-static,engine,vllm} ...  # see the module docstring
# Router policies on a mock fleet (CPU only; writes <out>.json and <out>.md, never overwrites):
python -m bench.router_compare --out results/router-compare-mock-x.json --label "..." \
    --replicas 4 --rate 30 --num-requests 600 --seeds 0,1,2 --mock-time-scale 2 \
    --shared-prefix-groups 64 --kv-cache-gb 0.25 --load-cap 1000 --approx-max-entries 256,1024
# --load-caps 4,8,16,32,1000 runs each prefix policy once per cap (a per-variant setting);
# --poll-interval sets how often the precise policy reads each replica's /cache
```

`README.md` has the user-facing version; `docs/gpu-runbook.md` is the ordered plan for a rented-GPU
session.

## 6. Known gaps, risks and gotchas

1. **The engine has run and been benchmarked on one GPU only** (a Capsule RTX 5090; D-019). The
   decode step was profiled (D-003): launch-bound, and the masked attention plus the gather take
   44 to 56% of GPU time at batch 32 to 64. The `--kv-cache-gb` sizing beyond 8 GB and other GPU
   classes are untested. Spend the first hour of any GPU session on a smoke test.
2. **The scheduler path against Hugging Face on a GPU** (`bench/scheduler_eval.py`): fp32 is
   50 of 50 identical in all three configurations including 35 forced preemptions; in bf16 some
   prompts diverge, and the logit-gap record (D-019) shows each divergence is a tie within one
   bf16 step. Prefix-cache **hits** have not yet been compared with an uncached run on real weights
   (a shared-prefix scheduler evaluation is planned).
3. **Prefill is not chunked**, so a long prompt delays decode for the others in that step.
4. **Random-token prompts share no prefixes**: without `--shared-prefix-*` the prefix cache shows a
   hit rate near zero. The shape of a shared-prefix workload is a modelling choice; state it.
5. **The mock's timing is a placeholder.** Calibrate it from real runs before using it to simulate
   or to compare policies.
6. **CI is green on the recent pushes** (GitHub Actions on Ubuntu with CPU torch: lint, type check
   and the CPU tests passed for `a5a5ea3`, `ad0215a` and `c7e416c`, checked with `gh run list`).
   Commits after `c7e416c` are local until pushed. CI does not run the GPU tests, the Docker builds
   or the Kubernetes checks.
7. **Windows-only test noise:** `test_runner.py::test_every_request_completes_in_order_at_its_scheduled_time`
   and `test_loadgen_cli.py::test_end_to_end_over_a_real_socket` fail intermittently because
   `asyncio.sleep` uses a 15.6 ms clock there. Not a code bug; the load generator runs on Linux for
   real measurements. A fresh `httpx.post` also costs ~0.2 s on Windows (new client, proxy lookup);
   tests share one client per scenario.
8. **Machine setup is not in the repo** (it is per laptop): a Python 3.12 virtualenv kept outside
   any OneDrive-synced folder, the Microsoft Visual C++ redistributable for torch, the weights
   downloaded by hand into `models/`, and an SSH deploy key (scoped to this repository, private key
   on the laptop only) behind the `origin` remote's host alias. If a push fails with "could not
   resolve hostname", the SSH config was lost again.
9. The change-log table at the end of `decisions.md` has only one row (the mock-fleet comparison);
   the A6000 baseline sweeps of D-015 are described in their decision entry, not in that table.
   (An earlier version of this file said `README.md` still read "Week 1 of 6"; it does not, and its
   status line is accurate.)
10. **The router's load cap (default 16 in flight per replica) disables prefix affinity once
    replicas hold that many requests**, which an overloaded or heavily loaded fleet does; see
    section 3. Choose it against the load, and always read the policies' `load_fallbacks` counter
    next to a hit rate: a hit rate that is the same for every policy usually means the cap, not
    the policies, decided.
11. **A short pilot hides saturation.** The load for the mock-fleet runs was first judged from
    200-request pilots, which looked fine; at 600 requests two of the three runs were overloaded.
    Judge a load from runs long enough to show the queue growing (achieved against offered rate).
12. **The mock-fleet result files record a commit and `dirty: true`**: the script was not committed
    when they were made. It is byte-identical to the committed version (checksums compared).
13. **Cleanup duty on shared Capsule machines (the user's requirement).** Everything the project puts
    on a Capsule machine lives in one directory, `~/pagedserve_work`, with every cache redirected
    into it (`env.sh` in that directory). Delete it only once the project is assumed finished with that
    machine (nothing there will be used again) or if the work is abandoned; while more runs are
    still plausible, keep it (it saves the 20-minute setup) and delete at the very end of the
    project. To delete: `rm -rf ~/pagedserve_work` on that machine, then check the home directory has only Capsule's own
    files (no `pagedserve*`, `.nv`, `.triton`, `.tilelang`, `~/.cache/uv`). Do not leave a GPU
    server running. The 2026-11-18/19 runs had left about 50 GB on five other machines (deleted on
    2026-11-23); the 5090 machine's directory is the only thing left to remove. Machine names are
    not recorded in this repository (D-017 convention): the user knows which machine it is, and the
    Capsule CLI lists it (`capsule list`); the work directory name identifies the files.
14. **The Capsule connection can drop** ("failed to establish connection"; it recovered within about
    20 minutes once). Work on the box runs detached with nohup and logs to files, so a dropped
    connection does not stop a sweep. `capsule exec` does not exit by itself: always wrap it in a
    timeout and run `capsule session endall` afterwards.

15. **The default open-file limit (1,024 per process) caps a saturated server at about 997 requests
    in flight.** The gather path's saturation runs hit it (170, 7, 153 and 82 of 1,920 requests failed
    with `ConnectError` in the last seconds of the arrivals); the faster paths never did. Found from
    the recorded results with `bench/inflight.py`; the limit itself was not read from the machine.
    Record and raise it (`ulimit -n`) for every system in any saturation comparison. In a steady-state
    window before the failures the kernel's gain over the gather path is 1.43 (1.33 to 1.54), against
    the pre-registered whole-run 1.55 (D-025).

## 7. Suggested next steps, in order

**First, the outstanding machine runs** (the D-027 rerun, the D-033 accuracy run with its four model
copies, D-028 and the open-file limit): the checklist with scripts and timings is at the top of
`docs/roadmap.md`. None has been run; they wait for the GPU machine.

1. ~~**Router policy comparison on a mock fleet.**~~ **Done** (`bench/router_compare.py`, five
   result files, D-007), including the load cap sweep and the poll interval test. Still needed
   before it counts as the CPU-only router gate: a Linux machine and more seeds (the Windows laptop
   shares one CPU between everything); the calibrated mock (below) so TTFT means something; the
   cap against another load or replica count; and a test of why the precise policy makes more
   no-match decisions than the approximate one (idea: log whether the prefix had been evicted).
2. ~~**Draft D-018**~~ **Drafted** in `docs/decisions.md` with status "draft": provider, instance,
   price, driver, budget and dates are `TO FILL`. Rental is planned for **Monday 2026-11-23**; the
   user has no provider or budget yet, so those are theirs to decide. Prices were recorded on
   2026-11-20 from the providers' own pages by an earlier session and **were not re-checked since**:
   RunPod H100 PCIe $2.89/h, Lambda H100 PCIe $3.79/h; AWS sells H100 only in 8-GPU sizes. The
   candidates, requirements (one GPU, a driver that advertises CUDA 13.2 or newer, the self-test
   passes, one GPU model for all three systems) and a rough budget (an estimate: about 5 to 7 hours,
   $15 to $27 at those prices) are in D-018. **Decisions for the user before renting:** provider and
   instance; a budget cap; and whether to run the Hugging Face baseline at batch 16, 32 and 64 on the
   new GPU (the runbook runs only 32; D-015 found the best setting at the edge of its grid on the
   A6000, and an 80 GB card lifts D-017's 16 GB limit), which adds about 24 runs. Until the
   provider is chosen, D-018 stays a draft and `docs/gpu-runbook.md` has no provider-specific steps.
3. **(Mostly done 2026-11-23, on the Capsule RTX 5090 of D-019; nothing was rented and D-018 stays a
   draft.)** Set up, self-test, GPU and CPU tests, smoke run, greedy evaluation and the three-way
   sweeps with four seeds, the shared-prefix runs and the monitored repeat are done and committed
   (`results/`, D-019, with pointers in D-005 and D-006). **Still to do:** the measured effects in
   D-003 and D-004 (paged cache and continuous batching; D-019 holds the comparisons), vLLM at
   higher rates and the engine's batch sizes beyond 64. Done since: the decode-step profile
   (D-003), the fp32 GPU contract (decided 2026-11-24, D-019) and the GPU scheduler-versus-Hugging
   Face evaluation (results in D-019 once recorded). **Next, in this order:** an independent review of the
   claims against the raw data, then the pre-registered CUDA-graph experiment (D-020), then delete the
   machine's directory. Original text of this step follows.
   **The GPU session itself**, following `docs/gpu-runbook.md`: set up, record both environments,
   smoke test, then the HF baseline, engine and vLLM sweeps with several seeds, then reports. Write
   the measured effects into D-003, D-004, D-005 and D-006 and add change-log rows. Plan as of
   2026-11-20: do the first hour (the smoke test) before the big sweeps, probably on Monday, on
   either a rented GPU or an Oxmiq Capsule machine; a Capsule machine is a candidate in D-018 and
   must meet the same requirements (dedicated, driver, self-test, disk).
4. **Speed work only after profiling.** Done (2026-11-24, opt-in, labelled diagnostics, caveats in
   D-025, D-020, D-026, D-027): the in-place decode-attention kernel and CUDA graphs on it, about 2.1
   times the gather path at saturation. **Next:** the phased plan in `docs/roadmap.md` (Plan from
   here, proposed after the second review): D-028 and the D-027 rerun in one machine session; then,
   without the machine, the credibility sweep and the Docker and Kubernetes files; then batched
   prefill (D-029) if D-028 triggers it, the matched same-session vLLM comparison (**D-030, planned
   2026-11-25**: five arms, a sharded load generator to build first, valid-run rules and the wording of
   the ratio fixed in advance; nothing built or run), what paging buys (D-023) and the router on real
   replicas (D-022); the kernel microbenchmark (**D-031, planned 2026-11-25**, optional, after D-030);
   stretch: the simulator and a second model. Chunked prefill (Sarathi-Serve) only if long prompts hurt decode latency.
5. **Deployment:** Dockerfiles and Kubernetes manifests in `deploy/` (cannot be tested without
   Docker or a cluster; mark them untested if written without one).
6. **Decide the D-011 extension** (serving simulator, which needs the calibrated mock, or a
   speculative-decoding study) once the router gate is met.

## 8. Index

- `docs/decisions.md`: D-001 to D-028 (D-018, the rented GPU, is a draft; D-019, the Capsule RTX
  5090, is in progress; D-020 CUDA graphs, D-025 the in-place kernel, D-026 and D-027 the bf16
  accuracy comparisons are run; D-021 attribution is run; D-022 to D-024 and D-028 are planned
  experiments), each with decision, reasons, alternatives and measured effect.
- `docs/roadmap.md`: the planned scope reconstructed (the original plan was lost) against what is
  done and what remains; a living document.
- `docs/second-review-2026-11-24.md`: a second review by a fresh chat given only the repository (its
  response unedited, with our verification notes and a tracker S2-1 to S2-18 mapping each finding
  to the roadmap item that handles it).
- `docs/independent-review-2026-11-24.md`: an independent, sceptical review of the claims against the
  raw data (unedited), with a status table of what has been done about each finding (a living
  document: update the table as findings are resolved).
- `docs/gpu-runbook.md`: the GPU session procedure; the `deploy/gpu_*.sh` scripts are the exact
  scripts that produced the RTX 5090 results (passes 1 to 4, profile, scheduler evaluation,
  ceiling, numerics).
- `results/`: write-once benchmark outputs (A6000 and RTX 5090 sweeps, greedy evaluations, the
  variation tables, and `router-compare-mock-*` for the mock-fleet comparison, which is not GPU
  data).
- Layout: `engine/` (model, KV cache, allocator, prefix cache, scheduler, server, mock), `router/`
  (policies, proxy, server), `bench/` (load generator, sweeps, baselines, GPU checks), `tests/`,
  `deploy/`.
