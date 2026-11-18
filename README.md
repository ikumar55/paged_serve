# pagedserve

A small LLM inference engine, written from scratch in Python and PyTorch, with a paged KV cache, continuous batching and prefix caching, plus a prefix-cache-aware router for multiple replicas.

**Status: early development (Week 1 of 6).** Nothing here has been benchmarked yet. Results will land in `results/` with the hardware, library versions, git commit and exact command recorded next to every number.

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
```

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

Design decisions, with the measured effect of each change, are tracked in [`docs/decisions.md`](docs/decisions.md).
