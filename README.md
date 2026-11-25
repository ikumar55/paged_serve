# pagedserve

A small LLM inference engine, written from scratch in Python and PyTorch, with a paged KV cache, continuous batching and prefix caching, plus a prefix-cache-aware router for multiple replicas.

**Status: working and measured on one GPU; the first speed work is done and opt-in.** The engine, server, prefix cache, router and mock backend are built and tested on CPU. The engine has been run and benchmarked on one RTX 5090 against Hugging Face static batching and vLLM (D-019): in bf16 it delivers about 1.6 times the throughput of a tuned Hugging Face static-batching server, and at saturation vLLM delivers about 4 times what it does (about 60 against 14 req/s, paired over four seeds), and vLLM with CUDA graphs disabled is still about twice the engine. Profiling the decode step on the real mix of sequence lengths shows the saturated step is GPU-bound, not launch-bound: attention and the gather of cached keys and values are 84% of GPU time, and the padded batch does about 3.7 times the work it needs, so the first speed work was attention that reads the cache blocks in place, with its success criteria written in advance (D-025). Two changes followed, both **opt-in and off by default** (so every earlier result still reproduces): a Triton decode-attention kernel (`--paged-attention-kernel`, D-025) and CUDA graphs on top of it (`--cuda-graphs`, D-020). In labelled diagnostic runs (four workload seeds, paired, interleaved in one session) saturated throughput rose from about 14 to about 22 req/s with the kernel and to about 29 req/s with graphs, about 2.1 times in total. vLLM, measured in an earlier session, saturates at about 60 req/s with 32 sequences and delivered at least 83 with 64 sequences (a lower bound, the load generator was saturated); the engine runs 64, so the like-for-like gap is at least about 2.9 times, not 2.1, and a same-session matched comparison is still to be done. **Read that with the caveats:** the bf16 correctness gates fixed in advance for both were missed by a near-tie and are recorded as failed in the decision log; fp32 output is identical to Hugging Face (50 of 50 prompts, including preemptions and prefix-cache hits), and a later accuracy comparison against fp32 found both as accurate as the original path in bf16, but that standard was chosen after the fact (D-026, D-027). The router has only been exercised on a simulated fleet (mock backend), never on real replicas. [`docs/STATUS.md`](docs/STATUS.md) says what is built, how it was checked and what is next. Results land in `results/` with the hardware and library versions recorded next to every number (the box ran a `git archive`, so the commit is in the decision log, not in each file).

## Layout

| Path | Purpose |
| --- | --- |
| `engine/` | Model forward pass, block allocator, scheduler, server |
| `router/` | Routing policies and streaming proxy |
| `bench/` | Load generator, baselines, environment capture |
| `tests/` | CPU-only tests (run in CI) and GPU tests (run by hand) |
| `deploy/` | Dockerfiles and Kubernetes manifests |
| `results/` | Benchmark outputs (generated, never edited by hand) |
| `docs/` | Design decisions and write-ups |

## Development

Requires Python 3.10 or newer.

```bash
make setup   # virtualenv and pinned dev dependencies
make check   # lint, type check and CPU tests
make test-real   # optional: compare against real Llama-3.2-1B-Instruct weights in models/
make test-gpu    # optional, needs a CUDA GPU; refuses to run on one that fails the self-test
python -m bench.gpu_selftest   # does this GPU return identical results for identical inputs?
```

The real-weights tests need the model files in `models/Llama-3.2-1B-Instruct/` (gitignored; download them yourself after accepting Meta's license) and skip themselves when they are absent.

## Load generator

Open-loop Poisson load against any OpenAI-style streaming `/v1/completions` server. It records time to first token, inter-token latency, end-to-end latency and tokens per second as JSONL, and plots them.

```bash
# On the machine that runs the server: record its hardware and library versions
python -m bench.env_capture --out results

# Send 60 s of 4 req/s load and record the results
python -m bench.loadgen run --url http://localhost:8000 --model MODEL --rate 4 --duration 60 \
    --label my-engine --server-env results/env-<commit>.json --out results/my-engine-rate4.jsonl

python -m bench.loadgen summarize results/my-engine-rate4.jsonl
python -m bench.loadgen plot results/my-engine-rate*.jsonl --out-dir results/plots
```

Result files are never overwritten, and the server address is redacted from the recorded command.

By default prompts are random token ids and share nothing. To exercise a prefix cache or a
prefix-aware router, add `--shared-prefix-groups 8 --shared-prefix-len 512 --shared-prefix-skew 1`:
each prompt then starts with one of eight fixed prefixes (the first ones more popular) followed by a
unique suffix described by the `--prompt-*` options.

## Engine server

The paged engine behind the same streaming API as the baselines, so the load generator drives either:

```bash
python -m engine.server --model-dir models/Llama-3.2-1B-Instruct --dtype bfloat16 \
    --max-batch-size 32 --kv-cache-gb 8 --port 8000
```

`--kv-cache-gb` sizes the pool of KV blocks (block size `--block-size`, default 16 tokens).
`--paged-attention-kernel` (GPU and Triton needed) reads each sequence's blocks in place instead of
gathering them into a padded tensor, and `--cuda-graphs` (needs the kernel flag) replays the decode
step as CUDA graphs; both are off by default and their status is in `docs/decisions.md`.
Full blocks are cached by content and shared between requests with a common prompt prefix; turn
that off with `--no-prefix-caching`. The scheduler admits requests while blocks are free and, when
the pool runs short, preempts the newest running request and recomputes it later. `/info` reports
batch sizes, preemptions, block use and prefix-cache hits. Greedy decoding and token-id prompts
only, as in the baseline. Benchmarked on one RTX 5090 (D-019). The attention path is the simple
gather-based one: every step rebuilds contiguous keys and values from the blocks, which a profile
shows is a large share of the GPU time at larger batches (D-003).

`--backend mock` serves the same API with no model: the real scheduler and block pool, with the
forward pass replaced by sleeps from a timing model (`--mock-*` options). It exists so the router,
CI and deployment work need no GPU. Its timings are placeholders, not measurements; do not quote
numbers from it as GPU results. `GET /cache` lists the content hashes of the blocks a replica
holds.

## Router

A streaming proxy in front of several engine replicas, with four policies: `round-robin`,
`least-outstanding`, `approx-prefix` (remembers where each prompt chunk went) and `precise-prefix`
(scores replicas by the blocks they report holding in `GET /cache`). It speaks the same API, so the
load generator can drive it:

```bash
python -m engine.server --backend mock --port 8001 &     # replicas; mock needs no GPU
python -m engine.server --backend mock --port 8002 &
python -m router.server --replica http://127.0.0.1:8001 --replica http://127.0.0.1:8002 \
    --policy precise-prefix --port 9000
curl 'http://127.0.0.1:9000/info?fleet=true'             # routing counters and fleet hit rate
```

Prefix policies need prompts that share prefixes; see `--shared-prefix-groups` in the load generator
section.

## Baselines

`bench.baselines.hf_server` serves a Hugging Face model with `generate` and static batching behind the same API, so the load generator can compare it with this engine and with vLLM:

```bash
python -m bench.baselines.hf_server --model MODEL_ID_OR_PATH --dtype bfloat16 \
    --max-batch-size 8 --batch-window-ms 20 --port 8000

python -m bench.loadgen run --url http://localhost:8000 --model MODEL --rate 4 --duration 60 \
    --fetch-server-info --label hf-static --server-env results/env-<commit>.json \
    --out results/hf-static-rate4.jsonl
```

Design decisions, with the measured effect of each change, are tracked in [`docs/decisions.md`](docs/decisions.md).

## Deploying

Dockerfiles, a compose file for a no-GPU demo stack and Kubernetes manifests (base plus two overlays) are in [`deploy/`](deploy/README.md). They have been validated offline (hadolint, kubeconform, and tests that parse every argument list with the servers' real parsers) and run: the images build, the mock stack runs under Docker Compose, and the `mock` overlay runs on a local `kind` cluster (service discovery, probes, security settings, self-healing and the GPU scheduling behavior were checked). The GPU path has never been run, because no GPU was available for it. `deploy/README.md` says what is verified and what to try first on a GPU machine.
