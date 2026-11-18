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
- **Measured effect:** n/a.

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
