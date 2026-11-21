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
- **Implementation (`router/policies.py`, `router/app.py`, `router/server.py`):** the router is a
  streaming proxy that relays a replica's response bytes unchanged, so the load generator drives it
  like one engine. Differences from the text above: the API takes token ids, so a "chunk" is a run
  of token ids (not characters) and is hashed with the engine's block hash chain, which is also what
  makes the precise policy's scores comparable to a replica's real block hashes; the precise policy
  polls each replica's `GET /cache` (default every 0.5 s), so what it knows is up to one poll old.
  Both prefix policies fall back to the least-loaded replica when the preferred one has `load_cap`
  requests in flight, and a block counts as a match only after the blocks before it. A replica that
  cannot be reached before its response starts is marked down for a cooldown and the request goes
  to another; a failure partway through a stream cannot be retried and reaches the client as a
  broken stream. A client that disconnects closes the upstream connection, so the engine aborts
  the request and frees its blocks. `GET /info?fleet=true` returns each replica's own `/info` and
  the fleet's prefix-cache hit rate.
- **Measured effect:** correctness only, against the mock backend of D-008 (so no number here says
  anything about a GPU). On twelve prompts in three shared-prefix groups over two replicas, the
  fleet's prefix-cache hit rate was 9 x 16 of 12 x 19 prompt tokens with the approximate policy,
  where every group stays on one replica, against 6 x 16 with round robin, which computes each
  group on both. That is the arithmetic of the example, checked by a test, not a benchmark of
  routing quality.

  **Policy comparison on a mock fleet** (`bench/router_compare.py`; three result files
  `results/router-compare-mock-*.json`, each with a `.md` report). This is the mock backend of D-008
  with placeholder timing, so it describes the routing policies and the plumbing and **says nothing
  about a GPU**. Setup common to the three runs: 4 mock replicas, with the replicas, the router and
  the load generator all on one Windows laptop; three workload seeds (0, 1, 2); 600 requests per
  run; shared prefixes of 512 tokens with unequal popularity (skew 1), unique suffix median 64
  tokens, output median 32; every variant gets the identical requests for a seed (checked by a
  digest stored per run) on a fresh fleet; no warm-up. Cells are the mean over the three seeds with
  the minimum to maximum in brackets, so the brackets are a spread and not a confidence interval.
  Three experiments, the third added after seeing the first two (below) to remove two problems found
  in them:
  - **A (ample cache):** 8 prefixes, 4 GB pool per replica, mock slowed 4x, 45 req/s, load cap 16.
    All four policies had the *same* fleet hit rate, 82.9%, and in every run exactly 290,816 hit
    tokens, which is 600 x 512 minus 32 x 512: each of the 8 prefixes computed once on each of the 4
    replicas, which is the worst case for a cache that never evicts (a policy that kept each prefix
    on one replica would compute each only once). Reason, from the policy counters: with the cap at 16 in flight the two prefix policies
    sent only about a tenth of their requests to the replica they believed held the prefix
    (approximate: 59 matched, 533 sent elsewhere because the holder was over the cap, 8 with no
    match, per 600), so they behaved like least-outstanding, and with a cache big enough for
    everything each policy then computed each prefix once on every replica. The fleet was also
    overloaded (about 34 req/s achieved of 45 offered), and the load generator sent late in one run.
    So this run says the cap, not the policy, decided; it does not say routing cannot help.
  - **B (tight cache):** 64 prefixes, 0.25 GB pool (about 512 blocks) per replica, otherwise as A.
    Hit rate: round-robin 43.9% (43.1 to 44.5), least-outstanding 43.9% (43.2 to 45.1), approximate
    46.3% (44.6 to 48.4), precise 43.6% (42.2 to 46.1). Only the approximate policy's range just
    clears round-robin's; the rest is within run spread. The fleet was badly overloaded (about 23.5 of
    45 req/s, 952 preemptions over all runs, TTFT p50 about 4 s for every policy), and the cap again
    sent about 84% of the approximate policy's requests elsewhere.
  - **C (tight cache, cap off, lighter load):** B's workload with the load cap at 1000 (never
    reached) and the mock slowed 2x at 30 req/s (about 28 req/s achieved, no preemptions). Hit rate:
    round-robin 42.2% (40.9 to 43.1), least-outstanding 43.7% (41.2 to 45.3), **approximate 70.9%
    (69.7 to 72.7), precise 67.8% (66.8 to 68.9)**. TTFT p50 78, 82, 64 and 66 ms, but TTFT p99 129,
    153, 143 and 140 ms (ranges overlap), so a lower median is shown and a better tail is not. The
    cost is concentration: computed-token imbalance (busiest replica against an even split) 1.06,
    1.11, 1.31 and 1.28, served-request imbalance 1.00, 1.02, 1.13 and 1.12, because requests follow
    the hot prefixes to the replicas that hold them.
  - **Memory limit of the approximate policy (in C):** unlimited 70.9%; limited to 1024 remembered
    chunks 64.2% (63.2 to 65.3); to 256 chunks 46.2% (44.0 to 48.4), no better than
    least-outstanding. The workload has about 2048 shared-prefix chunks plus unique suffix chunks.
  - **Drift of approximate from precise:** in C the approximate policy did not lag the precise one,
    it was about 3 points higher with non-overlapping ranges. A likely cause, **not tested**: the
    approximate policy records a prefix the moment it routes a request, while the precise policy
    learns of it at the next poll (every 0.5 s, about 15 requests at this rate), and the precise
    policy made 134 decisions with no match against the approximate policy's 61. Replica evictions
    (about 5,400 to 6,000 blocks per run in C) did not make the approximate policy worse than the
    precise one in these runs.
  - **Not established:** anything about a GPU, since all step times are placeholders; whether the
    TTFT differences (a few to a dozen milliseconds) are real; anything beyond this one workload
    shape and three seeds; behaviour on Linux. The result files record commit `1c8e080` with
    `dirty` true: the script was not yet committed when they were produced, and it is
    byte-identical to the version committed with them. (The load cap question is followed up
    below.)

  **Follow-up: load cap sweep and poll interval** (two more result files,
  `results/router-compare-mock-load-cap-sweep.*` and `results/router-compare-mock-poll-interval.*`).
  Same mock fleet as C (4 replicas, 64 prefixes of 512 tokens, 0.25 GB pool each, mock slowed 2x,
  30 req/s) with a longer horizon of 1200 requests and three seeds. Unlike A and B this load was
  not overloaded: 29.0 of 30 req/s achieved, no failed requests, no preemptions, the load generator
  never late. The cap is now a per-variant setting (`--load-caps`), applied to the two prefix
  policies; round-robin and least-outstanding have none. Hit rate, mean over seeds (range):
  - **The cap decides whether affinity happens, and the switch is sharp between 8 and 16.**
    Approximate / precise: cap 4 44.4% (43.7 to 45.7) / 44.5% (44.1 to 45.3), no better than
    least-outstanding's 44.1%, because 1125 of 1200 requests (approximate) went elsewhere; cap 8
    52.3% (52.1 to 52.6) / 54.9% (53.3 to 57.0); cap 16 71.5% (71.0 to 72.1) / 70.4% (69.3 to
    71.2); cap 32 71.8% / 70.6%; cap off 72.4% (71.1 to 73.9) / 70.4% (68.4 to 71.6). From 16
    upward nothing changed: at 16 the cap was reached for 9 of 1200 requests (approximate), so at
    this load the default of 16 is high enough. The A and B result was therefore a property of
    those overloaded runs, not of the cap value in general; the cap has to be set against the
    number of requests in flight, which depends on the load.
  - **No cap value gave both.** Computed-token imbalance was 1.07 (approximate) and 1.11 (precise)
    at cap 8, where hit rate was only 52 to 55%, and 1.32 to 1.44 (approximate) and 1.16 to 1.33
    (precise) at cap 16 and above, where it was 70 to 72%. The precise policy's 1.16 at cap 16
    looks lower than at 32 and off, but the ranges overlap (1.13 to 1.18, 1.23 to 1.46, 1.14 to
    1.48 for 16, 32, off) so that is not a finding. TTFT p50 was
    64 to 65 ms at cap 16 and above against 78 and 83 ms for round-robin and least-outstanding;
    p99 was 144 to 151 ms against 133 and 160 ms, overlapping, so again a better median is shown
    and a better tail is not.
  - **Precise polling every 0.05 s instead of 0.5 s** (cap off, same workloads, so the 0.5 s
    counterpart is the `cap=1000` row above): precise 71.9% (70.2 to 73.0) against 70.4% (68.4 to
    71.6), approximate 72.3% against 72.4%. The gap between approximate and precise narrowed from
    about 2.0 points to 0.4, but the precise policy's two ranges overlap, so with three seeds that
    is **not established**. The precise policy's no-match decisions barely moved (about 216
    against 232 per 1200, the approximate policy's being 64), so staleness of the poll does not
    explain most of them. Untested alternatives: the precise policy knows a prefix was evicted
    (the pool is tight) and then sends the request to the least-loaded replica, which can scatter
    a prefix, while the approximate policy stays loyal to the previous holder and recomputes there.
    Its TTFT p50 was also worse at the fast poll (73 ms, range 71 to 76, against 65 ms), possibly
    from the polling's own cost on a machine where everything shares the CPU; also untested.
  - **Still not established:** everything under "Not established" above, plus what the best cap is
    for another load or another number of replicas. These two files record commit `753130d` with
    `dirty` true, because the per-variant cap was not yet committed when they were produced; the
    script is byte-identical to the version committed with them.

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

### D-018 One benchmark machine, rented: a dedicated single GPU (DRAFT)

- **Status:** **draft, nothing is decided.** Provider, instance, price and driver are blank on
  purpose; fill every `TO FILL` when the instance is rented (planned from Monday 2026-11-23), then
  change this status to "decided". When decided, this supersedes the *machine choice* of D-017.
  D-017 itself stays as written: its validation of the software stack on the RTX 5060 Ti is still a
  true record.
- **Decision:** The Hugging Face baseline, this engine and vLLM are all measured on one rented,
  dedicated GPU, with one GPU model for all three systems.
  - Provider: `TO FILL`
  - Instance type and GPU model (the hardware name used in result labels, for example `h100`, never
    the provider's instance id): `TO FILL`
  - Price per hour, billing granularity, and the price of attached storage: `TO FILL`
  - Driver version and the CUDA version it advertises (must be 13.2 or newer): `TO FILL`
  - Budget cap in dollars and in hours: `TO FILL`
  - Rented at, released at (UTC): `TO FILL`
- **Why (reasons already in D-017 and D-015; nothing new has been measured for this draft):**
  D-017's machine is on a shared fleet (about 20 login sessions were open, so every batch of runs
  needs a contention check), and no benchmark has run on it, so switching discards no result. A
  dedicated instance takes other users off the GPU, the disk and the memory, which were the reasons
  the other four surveyed machines were rejected.
- **Requirements,** checked in this order, stopping at the first failure (procedure in
  `docs/gpu-runbook.md`):
  1. Exactly one GPU visible, with bf16 support and memory for the tuned Hugging Face batch
     (D-017 puts batch 64 at about 20 GB).
  2. The driver advertises CUDA 13.2 or newer, because the pinned torch is `+cu132`
     (`deploy/setup_gpu_box.sh` refuses otherwise; D-016 explains why a runtime newer than the
     driver is not acceptable). vLLM 0.34.0 has its own environment with torch 2.14.0+cu130.
  3. The D-016 determinism self-test passes on this instance (0 differing repeats), and its report is
     stored in every result file.
  4. At least 60 GB free disk, Python 3.12, Linux.
  5. Providers that hand out containers usually fix the driver on the host, so read the advertised
     CUDA version before paying for a whole session. (General knowledge about such providers, not
     checked for the candidates below.)
  If the instance is lost partway and replaced by a different GPU model, the comparison starts over
  or the two halves are labelled as different machines; they are never mixed.
- **Candidates.** Prices were recorded in `STATUS.md` on 2026-11-20 from the providers' own pages by
  an earlier session; **they have not been re-checked in this one, so look again on the day**.

  | Provider | Offer | Price per hour (recorded 2026-11-20) | To check before choosing |
  | --- | --- | --- | --- |
  | RunPod | H100 PCIe | $2.89 | Which driver and CUDA the host advertises; disk size and its price |
  | Lambda | H100 PCIe | $3.79 | Whether it is a full VM with a driver that meets requirement 2 |
  | AWS | H100 only in 8-GPU sizes (earlier session's finding) | not applicable | A single-GPU size would have to exist for this to be a candidate |
  | An Oxmiq Capsule machine (internal; raised by the user on 2026-11-20, nothing about it is known here) | unknown | unknown | Everything in the requirements: GPU model, a driver that advertises CUDA 13.2 or newer, dedicated for the whole session (D-017's shared fleet is the cautionary example), 60 GB of free disk, whether the self-test passes. The hardware owner can say which machines exist and who else uses them |

- **What this changes in the plan (open items, not decisions):**
  - D-017 ruled out Hugging Face batch 64 because 16 GB is too small; on an 80 GB card that no
    longer holds. D-015 found its best setting at the edge of the grid, and the best batch size on a
    faster GPU need not be 32. The runbook's baseline commands run only batch 32, so before 32 is
    called "tuned" on this GPU, batches 16, 32 and 64 should be run there.
  - D-017's caveat about hiding CPU overhead runs the other way on a faster GPU: this engine's
    Python work per step is a larger share of each step, so the engine's number is likely to look
    worse relative to vLLM here than on the 5060 Ti. That is a prediction to check, not a result.
  - H100 numbers must not be placed next to the A6000 numbers of D-015, or next to any other GPU's.
- **Rough budget (an estimate from the runbook's steps, not a measurement).** The runbook's sweeps
  are 12 runs for each of three systems (four seeds, three rates) plus about 12 shared-prefix runs
  with the prefix cache on and off, about 48 runs. Assuming 3 to 5 minutes per run (60 s of arrivals,
  draining at saturation, a fresh server each time; vLLM starts slower) that is about 2.5 to 4 hours,
  and setup, the smoke test and reports plausibly add 2 to 3 hours: 5 to 7 hours in all. At the two
  recorded prices that is about $15 to $27. Because the engine has never run on CUDA, the first
  session may be spent mostly fixing things, so a cap near twice that leaves room for one redo.
  The budget itself is the user's decision and is left as `TO FILL` above.
- **Caveats to state wherever its numbers appear:** one GPU class, so a headline comparison is not
  called general until repeated on a second class; a rented instance is dedicated to the GPU but
  its host's CPU and disk can still be shared, which the self-test and the load generator's lag
  flag do not rule out; no hostnames, addresses or instance ids in result files.
- **After the session:** the instance is stopped or terminated and attached storage deleted, and
  the provider's console is checked afterwards (`docs/gpu-runbook.md`, section 6).
- **Measured effect:** none yet.

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
| 2026-11-20 | Prefix-aware routing (approximate) against round-robin, mock fleet, tight cache, load cap off (D-007; **mock backend, uncalibrated timing, not GPU**) | Fleet prefix-cache hit rate, mean of 3 seeds | 42.2% (round-robin) | 70.9% (approximate) | `results/router-compare-mock-tight-cache-uncapped.json`, `python -m bench.router_compare ... --load-cap 1000 --mock-time-scale 2 --rate 30` (full command in the file) |
| 2026-11-20 | Load cap of the prefix policies, same mock fleet, 1200 requests at 30 req/s, not overloaded (D-007; **mock backend, uncalibrated timing, not GPU**) | Fleet prefix-cache hit rate of approx-prefix, mean of 3 seeds | 44.4% at cap 4 | 71.5% at cap 16 (unchanged, 71.8% at 32 and 72.4% with no cap) | `results/router-compare-mock-load-cap-sweep.json`, `python -m bench.router_compare ... --load-caps 4,8,16,32,1000` (full command in the file) |
