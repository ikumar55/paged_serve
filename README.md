# pagedserve

A small LLM inference engine, written from scratch in Python and PyTorch, with a paged KV cache, continuous batching and prefix caching, plus a prefix-cache-aware router for multiple replicas.

**Status:** working and measured on one GPU. The engine, server, prefix cache, router and mock backend are built and tested on CPU (the router only against a simulated fleet, never real GPU replicas). Everything below is on one RTX 5090 with Llama-3.2-1B-Instruct in bf16; the full evidence is in [`docs/STATUS.md`](docs/STATUS.md) and [`docs/decisions.md`](docs/decisions.md).

### Where it stands

Delivered requests per second at saturation (offered above capacity), mean of four workload seeds:

| System | req/s | What it is |
| --- | ---: | --- |
| Hugging Face static batching (its best setting) | 9.0 | the baseline most tutorials start from |
| This engine, default gather path | 14.0 | paged KV cache, continuous batching, prefix cache |
| + Triton decode-attention kernel (opt-in) | 21.6 | reads each sequence's blocks in place, no padding |
| + CUDA graphs on top (opt-in) | 29.0 | replays the decode step as a graph |
| vLLM, 32 sequences | 59.6 | measured in an earlier session |
| vLLM, 64 sequences | at least 82.6 | a lower bound: the load generator was saturated |

Profiling found the step GPU-bound on the real mix of sequence lengths (attention plus the gather were 84% of GPU time and the padded batch did 3.7 times the work it needed), which is why the kernel came first.

### Read these numbers with three caveats

1. **The speedups are labelled diagnostics, and two correctness gates were missed.** The bf16 gates fixed in advance for the kernel and for the graphs were missed by a near-tie and are recorded as failed (D-025, D-020). fp32 output is identical to Hugging Face (50 of 50 prompts, with preemptions and prefix-cache hits). Later accuracy comparisons against fp32 found both paths as accurate as the original in bf16, but those standards were chosen after the fact (D-026, D-027).
2. **The comparison with vLLM is rough.** vLLM and Hugging Face were measured in an earlier session, not interleaved with the later runs, and vLLM was at 32 sequences while the engine runs 64. The engine is at least about 2.1 times (32 sequences) to about 2.9 times (64 sequences) behind vLLM. A same-session matched comparison is planned.
3. **One GPU, one model, random-token prompts, four seeds, one run per cell.** The gather path hit a 997-request in-flight ceiling at the end of its runs, so the whole-run kernel gain of 1.55 times is 1.43 times in a steady-state window; the conclusions hold either way.

### Deploying

Dockerfiles, a compose file for a no-GPU demo and Kubernetes manifests are in [`deploy/`](deploy/README.md). They were validated offline and run: the images build, the mock stack runs under Docker Compose, and the mock overlay runs on a local `kind` cluster (service discovery, probes, security settings, self-healing and GPU scheduling behavior checked). The GPU path has never been run in a container.

Reproduce, if you have an NVIDIA GPU: `make setup`, download the weights into `models/` yourself (they are licence-gated), then see `docs/gpu-runbook.md` and `docs/STATUS.md` section 5 for the benchmark commands. Results land in `results/` with the hardware and library versions recorded next to every number.

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
only, as in the baseline. Benchmarked on one RTX 5090 (D-019). By default the attention path is the
simple gather-based one: every step rebuilds contiguous keys and values from the blocks, which a
profile shows is most of the GPU time on a real length mix (D-003, D-020); `--paged-attention-kernel`
removes that step.

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
