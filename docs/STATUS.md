# Project status and handover

A running record of what has been built, how it was checked, what is still open, and how to pick
the work up. **Update this file at the end of every step.** It complements
[`decisions.md`](decisions.md), which holds each design decision with its measured effect; where
the two disagree, `decisions.md` is the record and this file is the map.

Last updated: 2026-11-20, after the router policy comparison on a mock fleet and the D-018 draft
(steps 1 and 2 of section 7).

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

"Verified" means checked by the tests described, on the hardware named. Nothing has been run on a
GPU through the scheduler, the server or the router.

| Component | Code | Verified how | Not verified |
| --- | --- | --- | --- |
| Llama forward pass, contiguous KV cache (D-002, D-012) | `engine/model.py`, `config.py`, `weights.py`, `kv_cache.py`, `generate.py` | Bit-identical logits to Hugging Face in fp32 on CPU (50 prompts, a 2048-token prompt); within the 1e-3 target on an RTX A6000; greedy decoding identical on 200 of 200 prompts in fp32 and bf16 on the A6000 and in bf16 on the RTX 5060 Ti | Other GPUs; kernels other than plain PyTorch SDPA |
| Block allocator, paged KV cache (D-003) | `engine/block_allocator.py`, `paged_cache.py` | Paged logits exactly equal the contiguous cache's, with scattered blocks; deliberately broken variants fail | GPU; the gather-based attention path has never been profiled |
| Batched decode and scheduler (D-004) | `engine/scheduler.py`, `paged_cache.py` (`PagedBatchCache`) | Equals single-request greedy decoding on a tiny model, and equals Hugging Face `generate` on the **real weights**, fp32, CPU, 16 prompts x 24 tokens, including 9 preemptions by recompute (`tests/test_real_llama_scheduler.py`) | bf16, GPU, the 200-prompt evaluation through the scheduler, any speed |
| Prefix cache (D-005) | `engine/prefix_cache.py`, `block_allocator.py` | Tiny-model tests incl. eviction and preemption; six broken variants fail | Real weights, bf16, hit rates on real traffic; reused blocks may differ from an uncached run in rounding |
| Engine server | `engine/server.py` | Over real sockets on a tiny model: output equals reference, streaming, EOS, disconnects abort and free blocks, the load generator drives it | GPU; `--kv-cache-gb` sizing; throughput |
| Mock backend (D-008) | `engine/mock.py`, `server.py --backend mock` | Output is a pure function of the last token and position, so batching/preemption/routing cannot change it; pool accounting matches a real model | **Timing is uncalibrated placeholders**; only the decode step is loosely based on D-015 |
| Router (D-007) | `router/policies.py`, `app.py`, `server.py` | Mock replicas and router over real sockets: pass-through, the four policies, load cap, failover, 503, disconnect propagation, load generator through the router; seven of eight broken variants fail (the eighth is an equivalent mutant); `--max-entries` (the approximate policy's memory limit) reaches the policy from the command line | Real engine replicas, GPU, mid-stream replica death. The mock-fleet comparison below is not GPU evidence |
| Router policy comparison (D-007) | `bench/router_compare.py` | Replicas and router as real subprocesses over real sockets, mock backend; 29 deliberately broken variants of the script, the router flag and the policy factory each make a test fail. Writing them found one real flaw (a refused write could leave half a result behind, now fixed) and a test that the digest of the workload was recorded from what was meant rather than what was sent (now from what is sent); results in section 3 | Anything about a GPU; Linux; more than 3 seeds and one workload shape; the load cap's effect beyond the two settings tried |
| Load generator and workloads (D-014) | `bench/loadgen.py`, `workload.py`, `client.py`, `runner.py`, `metrics.py`, `results_io.py`, `plots.py` | Calibrated against a fake server; pinned hashes prove old workloads still regenerate exactly; optional shared-prefix mode | - |
| HF static-batching baseline (D-015) | `bench/baselines/`, `bench/sweep*.py` | Swept and recorded on an RTX A6000 (the only real benchmark numbers, section 3) | Not yet re-run on the chosen benchmark machine |
| Sweep tooling for engine and vLLM | `bench/sweep.py` | Command construction, validation and manifests tested with fake servers; the hf-static commands are pinned to what the recorded sweeps used | The vLLM server flags are from memory and unverified for vLLM 0.34.0; nothing has launched a real engine or vLLM sweep |
| GPU determinism self-test (D-016) | `bench/gpu_selftest.py` | Caught a faulty GPU once (results from it were retracted) | - |
| Rental-GPU setup | `deploy/setup_gpu_box.sh`, `docs/gpu-runbook.md` | Shell syntax only | **Never run on a real machine** |
| Dockerfiles, Kubernetes manifests | - | - | Not written; `deploy/` has only the setup script |

Test suite (CPU): `320 passed, 1 failed, 10 deselected` on Windows at this step (it was `299 passed,
2 failed` before it). The 10 deselected are GPU and real-weights tests. The failures are the load
generator's two timing tests, which fail intermittently on Windows only (one or both on any given
run; see section 6); they passed on the Linux machines in the log. The new tests add about 40
seconds on Windows, mostly from starting servers.

## 3. The only real measurements so far

All on an RTX A6000 (workstation GPU, not a datacenter one), Llama-3.2-1B-Instruct, bf16, one run
per cell unless stated; full method and caveats in D-015:

- **Tuned Hugging Face static-batching baseline:** max batch size 32, batch window 20 ms (batch 64
  is equivalent, 100 ms window indistinguishable): about **5.3 req/s (760 tokens/s) at
  saturation**, with 57% of decode slots wasted in the one run that measured it.
- **Run-to-run spread is 10 to 19%** of the mean (four seeds per cell in `results/variation.md`), so
  differences of that size mean nothing.
- Batch 16 saturates near 4.0 req/s (579 tokens/s); a batch window of 0 turns batching off.

D-017 chose one RTX 5060 Ti (16 GB) machine for the three-way comparison and validated the stack on
it (greedy evaluation: 200 of 200 identical, bf16), but **no benchmark has run there, and none has
run this engine on any GPU.** vLLM served the model once as a smoke test. Do not quote the A6000
numbers next to anything measured on another GPU.

D-016: one RTX PRO 6000 Blackwell returned different results for identical inputs; results from
it are retracted. Every GPU measurement must be preceded by the self-test.

### Mock-fleet router comparison (not GPU, timing uncalibrated)

Three runs of `bench/router_compare.py` (4 mock replicas and the router as processes on this Windows
laptop, 3 seeds, 600 requests each; details, counters and caveats in D-007, files
`results/router-compare-mock-*.json` and `.md`, every one labelled mock/uncalibrated). Do not quote
these as performance. What they do show, about routing and plumbing:

- **Default load cap (16) at a load near saturation turns prefix routing off.** In the two runs
  with it, the prefix policies sent about 85 to 90% of requests elsewhere because the holder was
  over the cap. With an ample cache all four policies then had identical hit tokens (A), and with a
  tight cache the hit rates were within run spread except a small edge for the approximate policy
  (B). Those two runs were also overloaded.
- **With the cap off and a lighter load (C), prefix-aware routing clearly raised the fleet hit
  rate:** approximate 70.9%, precise 67.8%, against round-robin 42.2% and least-outstanding 43.7%
  (three seeds, ranges do not overlap). TTFT p50 fell (78 to 64 ms in a placeholder-timed mock), p99
  did not, and work concentrated on the replicas holding the hot prefixes (computed-token imbalance
  1.3 against 1.06).
- **The approximate policy's memory matters a lot:** limited to 1024 chunks 64.2%, to 256 chunks
  46.2% (about least-outstanding). It did **not** lag the precise policy; it was higher by about 3
  points, probably because the precise policy only learns of a prefix at its next 0.5 s poll (an
  untested explanation).

The load cap therefore has to be chosen against the load (open item), and the comparison has to be
repeated on a calibrated mock and on real replicas before any of this is called a router result.

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
```

`README.md` has the user-facing version; `docs/gpu-runbook.md` is the ordered plan for a rented-GPU
session.

## 6. Known gaps, risks and gotchas

1. **The engine has never run on CUDA.** Likely trouble spots: the boolean attention mask in
   batched decode (could fall to a slow kernel), the indexed writes into the KV pool, bf16 ties
   (D-010), and `--kv-cache-gb` sizing. Spend the first hour of any GPU session on a smoke test.
2. **No GPU check of the scheduler against Hugging Face.** `bench/greedy_eval.py` tests the plain
   engine. In bf16 exact ties make token equality unreliable, so decide how to measure that before
   asserting it.
3. **Prefill is not chunked**, so a long prompt delays decode for the others in that step.
4. **Random-token prompts share no prefixes**: without `--shared-prefix-*` the prefix cache shows a
   hit rate near zero. The shape of a shared-prefix workload is a modelling choice; state it.
5. **The mock's timing is a placeholder.** Calibrate it from real runs before using it to simulate
   or to compare policies.
6. **CI has not been checked** since the recent pushes (GitHub Actions runs on Ubuntu with CPU
   torch). Look at the Actions tab; Linux-only failures are possible.
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
10. **The router's default load cap (16 in flight per replica) disables prefix affinity** once
    replicas hold that many requests, which a loaded fleet does; see section 3. Choose it against
    the load, and always read the policies' `load_fallbacks` counter next to a hit rate.
11. **A short pilot hides saturation.** The load for the mock-fleet runs was first judged from
    200-request pilots, which looked fine; at 600 requests two of the three runs were overloaded.
    Judge a load from runs long enough to show the queue growing (achieved against offered rate).
12. **The mock-fleet result files record a commit and `dirty: true`**: the script was not committed
    when they were made. It is byte-identical to the committed version (checksums compared).

## 7. Suggested next steps, in order

1. ~~**Router policy comparison on a mock fleet.**~~ **Done** (`bench/router_compare.py`, three
   runs, D-007). Follow-ups that would make it count as the CPU-only router gate: a load cap chosen
   against the load (try several, e.g. 16, 32, 64, off, at one load that is not overloaded); a
   longer horizon and a Linux machine; the calibrated mock (below) so TTFT means something; and a
   test of the untested explanation for precise trailing approximate (shorten `--poll-interval`).
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
3. **The GPU session itself**, following `docs/gpu-runbook.md`: set up, record both environments,
   smoke test, then the HF baseline, engine and vLLM sweeps with several seeds, then reports. Write
   the measured effects into D-003, D-004, D-005 and D-006 and add change-log rows.
4. **Speed work only after profiling:** CUDA graphs and removing Python overhead first (D-003), a
   fused paged-attention kernel only as a stretch, chunked prefill (Sarathi-Serve) if long prompts
   hurt decode latency.
5. **Deployment:** Dockerfiles and Kubernetes manifests in `deploy/` (cannot be tested without
   Docker or a cluster; mark them untested if written without one).
6. **Decide the D-011 extension** (serving simulator, which needs the calibrated mock, or a
   speculative-decoding study) once the router gate is met.

## 8. Index

- `docs/decisions.md`: D-001 to D-018 (D-018, the rented GPU, is a draft), each with decision,
  reasons, alternatives and measured effect.
- `docs/gpu-runbook.md`: the rented-GPU procedure.
- `results/`: write-once benchmark outputs (A6000 sweeps, greedy evaluations, the variation table,
  and `router-compare-mock-*` for the mock-fleet comparison, which is not GPU data).
- Layout: `engine/` (model, KV cache, allocator, prefix cache, scheduler, server, mock), `router/`
  (policies, proxy, server), `bench/` (load generator, sweeps, baselines, GPU checks), `tests/`,
  `deploy/`.
