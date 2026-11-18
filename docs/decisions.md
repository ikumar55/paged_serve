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

- **Status:** decided; license acceptance pending on the author's side.
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
- **Measured effect:** not yet measured.

### D-005 Prefix cache: block hash chain, refcounts, LRU eviction

- **Status:** decided.
- **Decision:** Hash each full block together with its parent block's hash. Share blocks by
  reference count. Evict unreferenced blocks in LRU order. Only full blocks are cacheable.
- **Why:** A hash chain makes a block's identity depend on its entire prefix, so equal hashes mean
  equal prefixes. Refcounts make sharing safe across running sequences.
- **Alternatives:** A radix tree over tokens (more precise sharing of partial prefixes, more
  machinery).
- **Measured effect:** not yet measured.

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

- **Status:** decided.
- **Decision:** The engine gets a `--backend=mock` mode driven by a calibrated timing model.
- **Why:** Router, Kubernetes and CI work should never need a GPU. The timing model is also the
  starting point for the optional serving simulator.
- **Measured effect:** n/a.

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
- **Measured effect:** n/a.

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

- **Status:** decided; real-model numbers pending (needs the model weights and the GPU).
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
- **Measured effect:** not yet measured on a real model. A smoke run against a tiny random-weight
  model on CPU (40 requests at 40 req/s, 30 ms batch window, 4 slots) gave a median TTFT of about
  35 ms, i.e. the batch window dominates at low load, and a mean batch size of 2.1. That checks the
  tooling, not performance.

## Open verification items

- Confirm the `LlamaConfig.llama_3_2_1b()` hyperparameters against the downloaded `config.json`
  once model access is granted. The preset was transcribed from the public model card.
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
