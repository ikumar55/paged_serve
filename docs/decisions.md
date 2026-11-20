# Design decisions

Each entry records what was decided, why, what else was considered, and the **measured** effect. The
measured-effect field stays "not yet measured" until a benchmark result file exists to cite; no
estimates are recorded here as if they were results.

## Principles

- **Correctness before speed.** Every optimization lands with a test showing that outputs did not
  change.
- **Profile before optimizing.** Each change and its measured effect is recorded in the change log
  at the bottom of this file.
- **Honest benchmarks.** Every result file records GPU model, driver, CUDA version, library versions,
  git commit and the exact command. Baselines run on the same model, dtype and maximum length, and
  results report where this engine is slower.
- **My own implementation.** Ideas come from public papers and documentation and are attributed
  below. No code is copied from vLLM, SGLang or other projects.
- **Public models and data only.** Weights and API tokens are never committed.

## Decisions

### D-001 Model: Llama-3.2-1B-Instruct, with TinyLlama as ungated fallback

- **Status:** decided; license accepted and the weights verified (see D-002 and D-010).
- **Decision:** Develop against Llama-3.2-1B-Instruct. Use TinyLlama-1.1B for bring-up if the
  Llama license is not yet accepted, and keep the model loader config-driven so a second
  architecture is cheap to add.
- **Why:** Llama 3.2 is widely recognized and small enough to benchmark on one GPU. It differs from
  older Llama checkpoints in two ways that silently break logit matching if missed: tied input and
  output embeddings, and Llama-3-style RoPE frequency scaling. The tiny test configs cover both.
- **Alternatives:** TinyLlama only (simpler, but less representative); Qwen3-0.6B (ungated, modern,
  but adds query/key normalization to the forward pass).
- **Measured effect:** n/a.

### D-002 Forward pass in plain PyTorch; Hugging Face only as a test reference

- **Status:** decided.
- **Decision:** Reimplement RoPE, grouped-query attention, RMSNorm and SwiGLU directly and load the
  public weights. Hugging Face model classes are used only in `tests/` and baselines.
- **Why:** The project is about the serving internals; wrapping a library model would hide them.
  Having a deterministic reference model in tests makes every later change checkable.
- **Measured effect:** On four tiny random-weight configurations (plain, tied embeddings,
  Llama-3 RoPE scaling, both), CPU, fp32, batch 3 by 21 tokens, the max absolute logit difference
  against the Hugging Face reference was 0.0 (bit-identical) in a local run. This is a
  tiny-model check only; the real-model comparison waits for the weights. The test enforces a
  bound of 1e-4, tighter than the 1e-3 target in D-010. As a sanity check that the tests can
  fail, deliberately breaking the model gave: RoPE scaling dropped 6.15, RoPE theta off by 1%
  0.043, RMSNorm epsilon 1e-5 to 1e-3 0.036 (max absolute logit difference).

  **Real model (Llama-3.2-1B-Instruct, fp32, CPU, local run, `make test-real`):** the max
  absolute logit difference against the Hugging Face reference was 0.0 (bit-identical) over 50
  prompts (10 sentences, 40 random token sequences of 4 to 32 tokens) and on one 2048-token
  prompt; greedy generation matched on 8 of 8 prompts over 24 tokens. The target in D-010 is
  1e-3. Exact equality reflects both implementations running the same operations in the same
  order on CPU in fp32; it should not be expected on a GPU, in bf16 or with custom kernels, where
  the 1e-3 bound is the contract. Sensitivity of these tests, measured by deliberately breaking
  the model: dropping RoPE scaling changes logits by 0.086 on a 32-token prompt and 0.78 on a
  2048-token prompt, and a 1% error in RoPE theta by 0.0054 and 0.028, all above 1e-3, which is
  why a long prompt is included. This is a small sample; the planned 200-prompt, 64-token
  evaluation is run by `bench/greedy_eval.py` (results below).

  **Correction: the GPU paragraph that follows is superseded and must not be cited.** The GPU
  it was measured on was later found to return different results for identical inputs. An
  elementwise `silu(x) * y` on unchanged inputs gave a different result in 1020 of 3000
  repeats, always in the same few rows; a whole-model forward pass differed by up to 0.45 in
  logit space between two identical calls; a float64 CPU reference put Hugging Face's own GPU
  output up to 0.9 away from the truth while its CPU output stayed within 3e-5. ECC counters
  were clean, so ECC did not catch it, and a single linear layer and the attention kernels
  passed in isolation. Whether the cause is that GPU or the CUDA 13.2 runtime on a driver that
  advertises CUDA 13.0 is not yet established. The figures below, the "noise floor" reading and
  the "bit-identical to Hugging Face" observation all assumed a GPU that computes the same thing
  twice, so none of them is evidence. They stay only so the record shows what was measured. The
  GPU checks will be re-run on a GPU that first passes a determinism self-test, and every GPU
  result file will record that self-test.

  **Superseded: real model on GPU (Llama-3.2-1B-Instruct, one NVIDIA RTX PRO 6000 Blackwell
  Workstation Edition with 96 GB, driver 580.178.04, torch 2.15.0+cu132, TF32 disabled,
  `make test-gpu`):**
  this is a workstation GPU, not a datacenter one, so nothing here transfers to an A100 or H100.
  In fp32 the max absolute logit difference against the Hugging Face fp32 CPU reference was
  2.0e-4 over the same 50 prompts and 9.2e-6 on the 2048-token prompt, inside the 1e-3 target.
  Hugging Face's own GPU output differs from its own CPU output by 2.0e-4 and 4.4e-5 on those
  prompts, so these differences are the GPU's numerical noise floor, not engine error. Against
  Hugging Face on the same GPU the engine's logits were bit-identical (max absolute difference
  0.0 over the 50 prompts). Both call the same kernels at these shapes, so that is an observation
  about this setup, not a guarantee, and the 1e-3 bound stays the contract. One run each; the
  prompts are the ones used for the CPU numbers above.

  **Real model on GPU, re-measured on a GPU that passed the self-test (D-016):** one NVIDIA RTX
  A6000 (48 GB, driver 580.126.20), torch 2.15.0+cu132, TF32 disabled. Again a workstation GPU,
  not a datacenter one. `make test-gpu` gave the same numbers on three runs. In fp32 the max
  absolute logit difference against the Hugging Face fp32 CPU reference was **7.9e-4** over the
  50 prompts (79% of the 1e-3 target) and 1.8e-5 on the 2048-token prompt; against Hugging Face on
  the same GPU it was exactly 0.0. The size of the first number is explained by a float64
  reference (`tests/fp64_truth_report.py`, largest logit 31.5): against it, Hugging Face fp32 on
  CPU is off by at most 3.7e-4 (median 2.6e-5), Hugging Face fp32 on this GPU by 4.2e-4 (median
  3.0e-5) and this engine on the GPU by 4.2e-4 (median 3.0e-5); on the 2048-token prompt 6.7e-6,
  1.8e-5 and 1.8e-5. So any fp32 implementation, CPU or GPU, is up to about 4e-4 from the truth,
  and two of them differ by up to about the sum, which is the 7.9e-4. This corrects the earlier
  reading of that difference as a property of the GPU. It also means the 1e-3 target leaves a
  margin of about 2.5 times the error of a single implementation, and that matching Hugging
  Face here means being as accurate as it is, not sharing its mistakes.

  **Greedy decoding, 200 prompts x 64 tokens** (`bench/greedy_eval.py`, commit d95f200, result files
  in `results/`): 80 prose and 60 code windows of this repository's own text plus 60 random-token
  sequences (the ids and a hash of the set are stored in each file), engine against Hugging Face
  `generate` on the same GPU and in the same dtype, EOS disabled so every output has 64 tokens.
  fp32: 200 of 200 outputs identical (95% interval 98.1% to 100%), the top-1 prediction agrees at
  all 12,800 teacher-forced steps, and the max logit difference over all of them is 0.0. bf16:
  200 of 200 identical, 12,800 of 12,800 steps agree, max logit difference 0.0. That is one
  model, one GPU and one prompt set, so it is a measurement and not a guarantee. Against the
  golden reference (Hugging Face fp32 on CPU, same prompts, engine in fp32 on the GPU): also 200
  of 200 identical and 12,800 of 12,800 steps in agreement; the max logit difference over all
  steps was 2.4e-4, consistent with the ~4e-4 fp32 error scale above, and only 2 of the 12,800
  steps had a top-2 gap small enough (at most twice that difference) for a flip to be possible,
  neither of which flipped. The GPU may be
  shared with other users (utilization of 12 to 23% appeared with no visible process of ours),
  which cannot affect these correctness results but would affect any latency measurement.

### D-003 Attention path: gather first, CUDA graphs before custom kernels

- **Status:** decided.
- **Decision:** Start with gather-based paged attention in PyTorch (correct, slow). Prefill stays
  gather/SDPA-based. Decode speed comes first from CUDA graphs and removing Python overhead. A
  Triton paged-attention decode kernel, tested against the gather version, is an optional stretch.
- **Why:** For a model this small at modest batch sizes, decode is dominated by kernel-launch and
  Python overhead, so CUDA graphs are expected to give the larger and lower-risk win. This is a
  hypothesis to confirm with profiling, not a result.
- **Measured effect:** not yet measured.

### D-004 Scheduler: iteration-level batching, admission by free blocks, preemption by recompute

- **Status:** decided.
- **Decision:** Waiting and running queues, one scheduling decision per forward pass, admission
  gated on free blocks, and preemption by recomputing the evicted sequence when the pool runs out.
- **Why:** Recompute avoids a swap path and CPU memory pool, keeping the first version small.
- **Alternatives:** Swapping KV blocks to CPU memory (more code, more failure modes).
- **Implementation (`engine/scheduler.py`, `engine/paged_cache.py`):** each step first makes room
  for every running sequence's next token, preempting the newest running sequence by recompute
  when the pool is short (it has produced the least, so recomputing it wastes the least); then
  runs one batched decode over sequences of different lengths (block tables gathered into a
  padded tensor with an attention mask); then admits waiting requests that fit and prefills them
  one at a time. A step that preempted does not admit, to avoid evicting again at once. A request
  that could never fit in the pool alone is refused when submitted, so the oldest running
  sequence can always finish. Prefill is not chunked, so a long prompt delays the decode of the
  others for that step.
- **Measured effect:** correctness only; nothing about speed or memory has been measured.
  Llama-3.2-1B-Instruct, fp32, on a CPU (local run, `tests/test_real_llama_scheduler.py`): 16
  prompts (6 sentences and 10 random-token sequences from the shared 50) of 24 new tokens each,
  compared with Hugging Face `generate` on the same CPU. With a pool large enough for the whole
  batch (batch size 8, no preemption) all 16 outputs were identical to Hugging Face's, token for
  token. With a pool of 14 blocks of 16 tokens, which cannot hold eight requests, the scheduler
  preempted 9 times over 70 steps and all 16 outputs were still identical, with no blocks
  leaked. On a tiny random-weight model the same checks pass with deliberately broken variants
  failing (unmasked padding, a lost token history, a block leak). This is a small sample on one
  CPU in fp32: bf16 (where D-010 notes exact ties break differently) and GPUs have not been run
  through the scheduler, and the 200-prompt evaluation of `bench/greedy_eval.py` has not been
  repeated through it.

### D-005 Prefix cache: block hash chain, refcounts, LRU eviction

- **Status:** decided.
- **Decision:** Hash each full block together with its parent block's hash. Share blocks by
  reference count. Evict unreferenced blocks in LRU order. Only full blocks are cacheable.
- **Why:** A hash chain makes a block's identity depend on its entire prefix, so equal hashes mean
  equal prefixes. Refcounts make sharing safe across running sequences.
- **Alternatives:** A radix tree over tokens (more precise sharing of partial prefixes, more
  machinery).
- **Implementation (`engine/prefix_cache.py`, `engine/block_allocator.py`):** a block's hash is a
  128-bit BLAKE2 digest of its parent's hash and its tokens, so a collision that would silently
  reuse the wrong keys and values is not a practical concern. A full block is registered when it
  fills, during prefill or decode. When its last holder releases it, it stays on an evictable list
  (least recently released first) with its keys and values intact; a later prompt that hashes to
  it takes it back, and when the free list runs out the oldest evictable block is reused.
  A prompt may reuse at most all but one of its tokens, whole blocks only, because the model needs a
  forward pass over at least the last token to produce the next one. Preemption by recompute (D-004)
  benefits too, since the evicted sequence's own blocks are usually still cached when it comes back.
  Eviction is by release order, not by last use, so a block that is hit again moves to the end
  only when its holder releases it. Off with `--no-prefix-caching` on the server, so it can be
  switched off for a comparison with a system that has it off.
- **Measured effect:** correctness and bookkeeping only; no speed or memory has been measured. On
  a tiny random-weight model on a CPU, outputs with the cache on equal single-request greedy
  decoding for shared prefixes (sequential and in one batch), a fully cached prompt, and under a
  pool small enough to force both eviction and preemption, with no blocks leaked. A
  deliberately broken variant fails the tests in each of six ways tried, including a hash that
  ignores its parent: without it, the prompt `[A, A]` reuses the block cached for the first `A` of
  `[A, B]` at the wrong position. Tokens reused from cache come from keys and values computed
  earlier, possibly in a different batch shape, so equality with an uncached run is expected to
  hold to rounding and not bit for bit; this was exact on the tiny fp32 model and has not been
  checked in bf16 or on the real weights. The hit rate on real traffic depends on the workload and
  is unmeasured.

### D-006 Baselines: Hugging Face static batching and pinned vLLM

- **Status:** decided.
- **Decision:** (1) Hugging Face `generate` behind a static-batching wrapper; (2) vLLM at a pinned
  version, reported both at defaults and with matching features.
- **Why:** The first shows the gain over naive serving; the second keeps the comparison honest.
- **Measured effect:** not yet measured.

### D-007 Router policies: round-robin, least-outstanding, approximate and precise prefix-aware

- **Status:** decided.
- **Decision:** Compare four policies. Approximate prefix-aware routing keeps a map from hashed
  prompt-prefix chunks to replicas (character chunks, so the router needs no tokenizer), picks the
  longest match, and falls back to the least-loaded replica above a load cap. Precise routing has
  each replica expose the block hashes it currently holds and scores replicas against that.
- **Why:** Approximate routing can drift from what replicas really cache (for example after
  eviction). Comparing both modes measures that drift directly.
- **Measured effect:** not yet measured.

### D-008 Mock backend for router, deployment and CI work

- **Status:** decided; implemented in `engine/mock.py` and `engine.server --backend mock`, with a
  timing model that is **not yet calibrated**.
- **Decision:** The engine gets a `--backend=mock` mode driven by a calibrated timing model.
- **Why:** Router, Kubernetes and CI work should never need a GPU. The timing model is also the
  starting point for the optional serving simulator.
- **Implementation:** the mock replaces only the model. The scheduler, block pool, prefix cache,
  preemption and HTTP server are the real ones, so a block is claimed, shared and evicted exactly
  as with a real model and tokens served from the prefix cache cost no prefill time. The mock
  sleeps for a prefill or decode duration from a `TimingModel` and returns a next token that is a
  pure function of the last token and its position, so output does not depend on batching,
  preemption or the replica, and a router that corrupts a stream is caught. The pool is sized as
  Llama-3.2-1B in bf16 would be for the given `--kv-cache-gb`, so memory pressure appears when
  it would on a real GPU of that size. The server also serves `/cache`, the content hashes of the
  blocks it holds, for the precise router policy of D-007.
- **Measured effect:** none, and the timing is a placeholder: only the 9 ms decode step is loosely
  based on the 9 to 11 ms inter-token latency recorded for the Hugging Face baseline on an A6000
  (D-015); the prefill figures are guesses. Anything measured against the mock describes the
  router and the plumbing, not a GPU, until the timing model is calibrated against real runs.

### D-009 Tooling and repository layout

- **Status:** decided.
- **Decision:** Top-level packages `engine/`, `router/` and `bench/` as laid out in the project
  skeleton. Python 3.12 for development. ruff (lint and format), mypy in strict mode for the
  packages, pytest for tests. CPU-only tests run in GitHub Actions; tests needing a GPU carry the
  `gpu` marker and are run by hand. Direct dependencies are pinned in `constraints.txt`; result
  files record the versions actually used.
- **Why:** One-command local checks (`make check`) that match CI keep every commit green.
- **Alternatives:** A `src/` layout with a single `pagedserve` package (cleaner imports, deviates
  from the planned directory structure).
- **Measured effect:** n/a.

### D-010 Correctness tolerances

- **Status:** decided; revisit after the first baselines.
- **Decision:** Check logits in fp32 against Hugging Face with a tight tolerance (target: max
  absolute difference within 1e-3). For bf16, report top-1 agreement and mean absolute difference
  instead of a tight absolute bound. For generation, report the greedy-token match rate over a
  fixed prompt set and the position of the first divergence.
- **Why:** bf16 has about three significant decimal digits, so representable values near magnitude
  8 to 16 are spaced roughly 0.06 apart. Different but equally valid accumulation orders will
  differ by more than a small absolute tolerance, so a tight bf16 bound would fail for reasons
  unrelated to bugs. One early token mismatch also cascades through the rest of a sequence, so the
  divergence position is more informative than a single match rate.
- **Measured effect:** fp32 on the real model is covered under D-002. For bf16 (this engine in
  bf16 against the Hugging Face fp32 reference, 50 prompts, CPU, local run): top-1 token
  agreement 0.9614 across all positions and mean absolute logit difference 0.0400. These
  describe how far bf16 drifts from fp32 and are reported, not bounded, as decided above.
  The same test on a Linux x86 CPU gave 0.9710 and 0.0384, so the figure moves by about 0.01
  with the hardware's math libraries alone. Superseded, see the correction under D-002 (the GPU
  below returned inconsistent results): on the GPU named there (same 50 prompts, one run): this
  engine in bf16 had top-1 agreement 0.9626 and mean absolute difference 0.0381, and
  Hugging Face in bf16 had 0.9602 and 0.0367. The gap between them is about two tokens' worth
  and smaller than the hardware-to-hardware movement above, so it is not evidence that either
  is better.

  Re-measured on the RTX A6000 (D-002, same 50 prompts): this engine in bf16 had top-1 agreement
  0.9602 and mean absolute difference 0.0402, and Hugging Face in bf16 had 0.9626 and 0.0421,
  again too close to rank. A bf16 effect that matters for comparisons shows up in greedy
  decoding: on 217 of 12,800 steps (1.7%) the two best logits were exactly equal, and at 112
  steps Hugging Face's own `generate` (KV cache) and its own full forward pass broke the tie
  differently, while this engine matched Hugging Face's forward pass at every step (max logit
  difference 0.0). So bf16 greedy output is not well defined at exact ties. Comparing systems in
  bf16 therefore has to break ties the same way or report them, and `bench/greedy_eval.py` now
  counts them. An earlier version of that tool charged Hugging Face's self-inconsistency to the
  engine (107 spurious "disagreements" on a different prompt set); that was a flaw in the metric,
  fixed before any result was published.

### D-011 One extension, chosen after the Phase 2 gate

- **Status:** open (decision point at the end of Week 4).
- **Decision:** Default candidate is a calibrated serving simulator validated against the real
  multi-replica results; the fallback is a speculative decoding study. Neither starts before the
  router gate is met.
- **Why:** Keeps the core scope intact and picks the extension with evidence about schedule risk.
- **Measured effect:** n/a.

### D-012 Forward-pass implementation notes

- **Status:** decided.
- **Decision:**
  - RoPE cos and sin tables are computed in float32 by a plain object, not an `nn.Module`, so
    casting the model to bf16 cannot downcast the rotary frequencies. A test checks this.
  - Prefill with cached tokens builds an explicit causal mask over (cached + new) positions;
    decode uses no mask, and a prompt with no cache uses SDPA's built-in causal flag.
  - Grouped-query attention uses SDPA's native grouped-query support, with no key/value copies.
  - The weight loader refuses partial loads (missing, unexpected or mis-shaped tensors) and
    accepts tied checkpoints that omit `lm_head.weight`.
- **Why:** Each of these is a way a Llama 3.x port can silently produce plausible but wrong
  output.
- **Measured effect:** n/a (correctness only).

### D-013 Reference-comparison pitfalls

- **Status:** decided; applies to all later correctness tests and baselines.
- **Decision:** When comparing generation with Hugging Face, always pass an explicit attention
  mask. Treat Hugging Face's own full-recompute greedy loop, not just `generate()`, as the ground
  truth when investigating a mismatch.
- **Why:** In a test run, `generate()` disagreed with both this engine and Hugging Face's own
  uncached logits on a 30-token prompt. Cause: without an attention mask, `generate()` infers one
  from `pad_token_id` and silently masks every prompt token equal to that id. A test harness
  bug, not an engine bug. With an explicit mask, all prompts match. This also confirms that
  mismatches should be investigated at the first diverging position (top-two logit gap, mask,
  dtype) before being counted as engine errors.
- **Measured effect:** n/a.

### D-014 Load generator design

- **Status:** decided.
- **Decision:**
  - **Open loop.** Requests are sent at their Poisson arrival times whether or not earlier ones
    finished. A slow server then shows queueing delay instead of silently throttling the load.
    Every result records how late it was sent; if p99 lag exceeds 50 ms the run is flagged as
    generator-limited, because its latencies are understated.
  - **Seeded, independent streams.** Arrivals, prompt lengths, output lengths and token contents
    use separate random streams from one seed, so changing one distribution does not reshuffle
    the others and any workload can be regenerated exactly.
  - **Token-id prompts, forced output length.** Prompts are lists of token ids (exact lengths, no
    tokenizer) and requests set `ignore_eos`, so the output-length distribution is what was asked
    for rather than whatever the model decides. Lengths are lognormal with a median, a sigma and
    clipping bounds.
  - **Optional shared prefixes.** With `--shared-prefix-groups` and `--shared-prefix-len`, every
    prompt is one of a fixed set of random prefixes followed by a unique suffix, and the
    `--prompt-*` options then describe the suffix. Which prefix a request gets is drawn with a
    popularity proportional to `1 / (rank + 1) ** skew` (`--shared-prefix-skew`, 0 for uniform).
    This exists because random-token prompts share nothing, so they cannot show what a prefix
    cache or prefix-aware routing does. The shapes (how many prefixes, how long, how unequal) are
    modelling choices, not measurements of real traffic. The two extra random streams leave the
    four original ones untouched, and a test pins the output of the old workloads to hashes taken
    before the change, so recorded results can still be regenerated from their stored
    configuration.
  - **OpenAI-style streaming completions.** The same client drives this engine, vLLM and, later,
    the router. Token counts come from the server's `usage` when present; otherwise stream events
    are counted and the result is marked as such.
  - **Results are write-once JSONL** (meta line, one line per request, summary line), with an
    explicit schema version. The writer refuses to overwrite an existing file.
  - **Publishable by construction.** The server address is redacted from the recorded command,
    the home directory is replaced with `~`, and client errors keep only the exception type. A run
    without `--server-env` (the environment capture from the GPU machine) prints a warning that
    the result must not be published, because the client machine's hardware is not the server's.
  - Latency statistics cover successful requests only; failures are counted by cause. Percentiles
    use linear interpolation and every distribution reports its sample count.
- **Why:** These are the usual ways a load test quietly flatters the system under test
  (coordinated omission, retries hidden in averages, unreproducible workloads) or leaks details
  that should not be published.
- **Measured effect:** Calibration of the tool itself, not of any engine. Against a local test
  server configured with a 50 ms first-token delay and 20 ms between tokens, one local run
  measured TTFT p50 53.8 ms and inter-token p50 20.6 ms (60 requests), so the client plus local
  socket overhead is on the order of a few milliseconds. Offered 20 and 40 req/s, it sent on
  schedule (no generator-saturation flag).

### D-015 Hugging Face baseline: static batching behind the same API

- **Status:** decided; real-model sweeps recorded below (batch sizes up to 64, one run per
  configuration except where noted). Repeated runs to measure run-to-run variation are the last
  step before these count as the final baseline numbers.
- **Decision:** Baseline (1) is `model.generate` behind a small FastAPI server
  (`bench.baselines.hf_server`) that speaks the same streaming `/v1/completions` as everything
  else. Requests queue FIFO; a worker takes up to `max_batch_size` of them (waiting at most a
  batch window after the first), left-pads the prompts, and runs the whole batch to completion.
  A request that arrives mid-batch waits for the next batch, and every sequence holds its slot
  until the longest in its batch ends. Tokens stream step by step through a `generate` streamer, so
  TTFT includes queueing, the batch window and prefill but not the rest of the batch.
- **Why:** This is what a competent person would build with the standard library, and its two
  costs (waiting for the next batch, wasted slots) are exactly what continuous batching removes.
  The server reports both: `/info` includes batch-size counts and the fraction of decode slots
  spent on already-finished sequences, and the load generator records it before and after a run.
- **Caveats to state wherever its numbers appear:**
  - Greedy decoding and token-id prompts only; token events carry the id rather than detokenized
    text, which slightly favors this baseline over servers that detokenize.
  - `max_batch_size` and the batch window are tuning knobs. Before any comparison they must be
    swept and the best setting reported alongside the others, not a single arbitrary choice.
  - Single GPU, one batch at a time, bf16 on the GPU box.
- **Bug found while building it:** passing `eos_token_id=None` in the generation config does not
  disable EOS stopping; `generate()` fills in the model's default, so requests that ask to ignore
  EOS still stopped early and fell short of their forced output length. Fixed by passing an id
  outside the vocabulary, which can never be generated. A test guards this, because the load
  generator's forced output lengths depend on it.
- **Measured effect:** a sweep of 60 runs, `results/hf-static/` (table and per-run files in
  `REPORT.md`, plot in `ttft-p50.png`). Setup: Llama-3.2-1B-Instruct in bf16 on one RTX A6000
  (a workstation GPU, not a datacenter one), server and load generator on the same machine over
  localhost, commit 99b2cf1, GPU self-test passed (D-016). Open-loop Poisson arrivals, seed 0,
  lognormal prompt lengths (median 256, at most 2048) and output lengths (median 128, at most
  512, EOS ignored), 60 s of arrivals per run, a 120 s per-request deadline, a fresh server for
  every run. Max batch size 1, 4, 8 and 16; batch window 0, 20 and 100 ms; 0.5, 1, 2, 4 and 8
  requests per second. All 60 runs completed, none was contended by another GPU user and the load
  generator never fell behind. **One run per configuration**, so each run has 30 to 480 requests
  (30 at the lowest rate, so its p99 rests on 30 samples) and run-to-run variation is unmeasured.
  - **A window of 0 ms turns off batching.** The mean batch size was exactly 1.0 at every max
    batch size, so throughput stayed at about 0.7 req/s (105 tokens/s), the same as max batch size
    1. A baseline configured that way would look about 5 times slower than a tuned one.
  - **With a window of 20 ms or more, throughput grows with batch size.** At the heaviest load
    where every request still completed: 0.73 req/s (110 tokens/s) at batch 1, 1.60 (236) at
    batch 4, 2.63 (387) at batch 8 and 3.76 (566) at batch 16. The largest load each setting kept
    up with (throughput at least 90% of the arrival rate) was 1, 2 and 4 req/s for batch 4, 8 and
    16. The best setting is at the edge of the grid, so batch sizes above 16 must be tried before
    this counts as a tuned baseline.
  - **Slots are wasted.** The fraction of decode slots spent on sequences that had already
    finished was 36% at batch 4, 44% at batch 8 and 53% at batch 16 at those loads, because every
    sequence waits for the longest in its batch. This is the cost continuous batching removes.
  - **Latency while keeping up.** Batch 16 with a 20 ms window had TTFT p50 / p99 of 0.32 / 2.3 s
    at 0.5 req/s, 0.72 / 3.3 s at 1, 1.5 / 4.0 s at 2 and 2.4 / 7.8 s at 4 req/s. Inter-token
    latency was 9 to 11 ms in every run. At 8 req/s every configuration was overloaded (best p99
    TTFT 65 s; at batch 1, 80% of requests hit the deadline).
  - **Windows of 20 and 100 ms were indistinguishable** at this sampling (for example capacity of
    3.76 and 3.77 req/s at batch 16), but with a single run each that is a statement about this
    sample, not a finding.
  The "sustained" label uses a 90% cutoff that is a convention: `b16-w20-r4` (90.3%) and
  `b16-w100-r4` (88%) are not meaningfully different.

  **Extension to batch sizes 32 and 64, and repeated runs.** The first sweep's best setting was
  at the edge of its grid, so batch sizes 32 and 64 were added (`results/hf-static-ext/`, 12 runs,
  windows of 20 and 100 ms, 2, 4 and 8 req/s). Batch 16, 32 and 64 at a 20 ms window were then
  repeated at 4 and 8 req/s with three more workload seeds (`results/hf-static-seed{1,2,3}/` and
  `results/hf-static-b64-seed{1,2,3}/`), giving four runs per cell; `results/variation.md` has
  the table. Same setup and method throughout, none of the runs was contended. The self-test
  report in the `hf-static-seed*` folders is the one from the extension run about 25 minutes
  earlier; the batch-64 repeats ran their own.

  **The repeats show a run-to-run spread of 10 to 19% of the mean throughput**, so a single-run
  difference of that size means nothing. They also correct a conclusion I drew from the
  single-run extension, that batch 64 is worse than batch 32 (4.66 against 4.95 req/s at 8 req/s):
  with four runs each it is not (5.31 against 5.26), and that first reading was the low end of
  batch 64's spread. At 8 req/s offered (mean, with minimum to maximum of the four runs):
  batch 16 reached 4.02 req/s (3.76 to 4.16, 579 tokens/s), batch 32 5.26 (4.95 to 5.67, 757) and
  batch 64 5.31 (4.66 to 5.65, 764). Going from batch 16 to 32 therefore raises saturated
  throughput by about 31% with non-overlapping ranges, and going from 32 to 64 adds nothing
  measurable. TTFT p99 at that load was 57, 30 and 30 s (the ranges for 32 and 64 overlap), and
  TTFT p50 was 14 s at batch 32 and 10 s at batch 64 (ranges 12.4 to 16.6 s and 9.2 to 12.9 s).
  At 4 req/s batches 16, 32 and 64 are tied on throughput (3.65, 3.78 and 3.73 req/s).

  **The baseline to compare against** is therefore a max batch size of 32 (64 is equivalent) with
  a 20 ms window (100 ms was indistinguishable): about 5.3 req/s (760 tokens/s) at saturation on
  this GPU and workload, with 57% of decode slots wasted in the single run that measured it. The
  continuous-batching engine should be compared with that figure, not with the 3.76 req/s of the
  first single-run grid. The single-run inter-token latency at batch 64 was 24 ms against 14 ms at
  batch 32; this was not repeated.

### D-016 Every GPU measurement is preceded by a determinism self-test

- **Status:** decided; implemented in `bench/gpu_selftest.py` and enforced by `bench.greedy_eval`
  and `make test-gpu`.
- **Decision:** Before measuring on a GPU, repeat elementwise operations and a matrix multiply on
  identical inputs and require bit-identical results, and check `silu` and a matrix multiply
  against a float64 CPU result. The check fails closed (an explicit `--skip-selftest` exists and is
  recorded), and the report is stored in every result file.
- **Why:** The first GPU used returned different results for identical inputs and nothing in the
  usual checks said so (see the correction under D-002). The only symptom was unexplained logit
  differences between this engine and Hugging Face of up to 1.8, while greedy decoding agreed on
  200 of 200 prompts, so token agreement alone would have passed it.
- **Measured effect (a finding about one machine, not a benchmark):** on one NVIDIA RTX PRO 6000
  Blackwell Workstation Edition (driver 580.178.04), with torch 2.15.0+cu132 and again with torch
  2.14.1+cu130, repeating an operation on identical input gave different results: `silu` on
  3000 of 3000 repeats (about 6 wrong elements each, always within the same 8 of 189 rows),
  `sigmoid` on 300 of 3000, `exp` on 5 of 3000, while multiply, add, divide, `tanh`, `rsqrt`,
  `sin`, `log` and a mean never differed. ECC counters were zero, a 24 GB memory pattern test
  (3 passes) found no errors, and a single matrix multiply and the attention kernels passed in
  isolation. `silu` was off from the float64 result by a relative 0.21 in one run. Bisecting a
  whole-model forward pass with module hooks found the first differing output in the MLP of layer
  0, in 4 of 189 token rows. The pattern (the exponential/reciprocal path, the same rows,
  different values on every repeat, two software builds) points at a hardware fault, not at this
  engine or PyTorch, but that is an inference: it has not yet been reproduced on a second GPU
  or ruled out for that GPU with a different driver. Results from that machine are discarded.
  The same software (torch 2.15.0+cu132) on an RTX A6000 passed every check in three runs: 0 of
  300 differing repeats for each operation and a `silu` error against float64 of 1.0e-7, which
  makes a software cause much less likely.

### D-017 One benchmark machine for all three systems: an RTX 5060 Ti (16 GB)

- **Status:** decided; the stack is validated on it (below). No benchmark has been run on it yet.
- **Decision:** The Hugging Face baseline, this engine and vLLM are all measured on one machine,
  one NVIDIA GeForce RTX 5060 Ti (16 GB, Blackwell consumer, driver 580.126.20, 32 CPU cores,
  503 GB RAM, Ubuntu 24.04). The tuned Hugging Face baseline of D-015, measured on an RTX A6000,
  is re-run there; the A6000 numbers stay as a record but are not compared with anything measured
  on this GPU.
- **Why this machine:** five NVIDIA machines were surveyed (GPU memory and utilization sampled six
  times, other users' processes, free disk, cores, network, then the D-016 self-test, 3 runs each).
  Only this one met every hard criterion: it passed the self-test (0 of 300 differing repeats for
  every operation), its GPU was empty and idle, it had a terabyte of free disk and fast uploads
  from the workstation (about 19 MB/s for the weights, against about 1 MB/s to two others). The
  others failed on a resource, not on health (one also failed the self-test, see D-016):
  - The A6000 machine: 8.9 GB free disk on a full filesystem. vLLM 0.34.0 cannot share the engine's
    environment (a dry run shows it replaces torch 2.15.0 with 2.14.0 and CUDA 13.2 with 13.0), so
    it needs its own environment (measured here at 8.1 GB, next to 5.5 GB for ours).
  - An RTX PRO 4500 machine (32 GB): another user's idle vLLM server held 23.3 GB of the GPU's
    memory, leaving 8.8 GB, less than the tuned Hugging Face batch of 32 needs (about 10 GB), and
    it could become active at any time.
  - An RTX 5090 machine (32 GB): idle GPU, but a 99% full disk that shrank from 18 to 12 GB during
    the survey (other users writing), which cannot hold both environments plus the weights.
  - Not considered: Turing GPUs and T4s (no bf16), an 8 GB card (too small), ARM unified-memory
    machines (unrepresentative).
- **Caveats to state wherever its numbers appear:**
  - Consumer GPU with 448 GB/s of memory bandwidth, the lowest of the machines surveyed (the A6000
    has 768 GB/s, the 5090 about 1.8 TB/s; published specifications, not measured here), so absolute throughput is lower than on the A6000.
    All systems share the GPU, so comparisons between them hold; absolute numbers do not transfer.
  - A slower GPU hides CPU overhead: this engine is Python, and on a faster GPU its per-step
    scheduling cost would be a larger share of each step. A headline comparison should therefore be
    repeated on a second GPU class before it is called general.
  - 16 GB rules out Hugging Face batch 64 (about 20 GB), which D-015 found equivalent to batch 32.
  - The fleet is shared (about 20 login sessions were open); the load generator flags GPU contention
    and the GPU is checked before every batch of runs.
- **Validation on this machine (commit 7f2de3c, clean tree):** torch 2.15.0+cu132 and transformers
  5.23.0 in the engine environment; vLLM 0.34.0 with torch 2.14.0+cu130 in its own. The 176 CPU
  tests and the 3 GPU tests against the real weights passed. `bench/greedy_eval.py` in bf16 (engine
  against Hugging Face, same GPU, 200 prompts x 64 tokens, result in
  `results/greedy-eval-5060ti-bf16.json`): 200 of 200 identical, teacher-forced top-1 agreement
  12800 of 12800, max logit difference 0.0. This is the same agreement the A6000 gave. vLLM served
  the model (bf16, max model length 4096, 85% GPU memory use, about 13 GB in use) and accepted
  token-id prompts; one 8-token greedy completion was a smoke test only, not a measurement.
- **Correction recorded:** my earlier note that vLLM 0.34.0 resolves to torch 2.15.0 was wrong; it
  pins torch 2.14.0, which is why it needs a separate environment. A first run of the greedy
  evaluation on this machine was labelled with the machine's id, which ended up inside the result
  file; it was discarded and re-run under a hardware label (the leak scan caught it).

## Open verification items

- ~~Confirm the `LlamaConfig.llama_3_2_1b()` hyperparameters against the real `config.json`.~~
  Done: the preset equals the published config (checked by `test_preset_matches_the_published_config`).
- Revisit the throughput and latency targets after the Week 1 baselines.

## Attributions

Design ideas taken from public papers and documentation; no code was copied.

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention":
  block-based KV cache with per-sequence block tables.
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models":
  iteration-level scheduling.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs"
  (RadixAttention): reuse of cached prefixes across requests.
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve":
  chunked prefill.
- vLLM's publicly documented automatic prefix caching: hashing each block together with its
  parent's hash.
- llm-d's public documentation on prefix-cache-aware routing: the distinction between approximate
  (traffic-based) and precise (replica-reported) cache awareness.

## Change log with measured effects

| Date | Change | Metric | Before | After | Result file and command |
| --- | --- | --- | --- | --- | --- |
| | _No entries yet._ | | | | |
