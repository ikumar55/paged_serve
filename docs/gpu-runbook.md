# Runbook: a measurement session on a rented GPU

For one dedicated cloud GPU, billed by the hour. The aim is to spend the paid time measuring, not
installing. Do the steps in order and stop at the first failure.

**Status of this document:** nothing on the GPU path has been run yet. The engine has never run on
CUDA; the setup script, the vLLM server command and the engine and vLLM sweeps are tested only
with fake servers on a CPU. Treat the first hour as a smoke test (section 3), and expect to fix
something.

## 0. Before renting

- One GPU only, with a driver that supports the CUDA runtime of the pinned torch build (the
  `+cu132` wheel needs a driver that advertises CUDA 13.2 or newer). `deploy/setup_gpu_box.sh`
  checks this and refuses to continue otherwise (D-016).
- At least 60 GB of disk (two environments of 5 and 8 GB, weights, results) and Python 3.12.
- The comparison needs the Hugging Face baseline, this engine and vLLM on the **same** GPU model.
  Do not mix machines inside one comparison.
- Write the choice down as D-018 in `docs/decisions.md` (it supersedes the machine in D-017).

**Secrets.** Never copy the laptop's SSH private key to the rented machine. Never put a Hugging
Face token in a file in the repo or in a command that is recorded. Get the code onto the machine by
copying it from the laptop (`git archive HEAD | ssh <box> tar -x -C paged_serve`, or `scp`), not
by logging in to GitHub from the box. Result files must not contain hostnames, IPs or instance
ids: use a hardware name such as `h100` for labels, not the provider's instance id.

## 1. Set up the machine

```bash
cd paged_serve
VLLM_VERSION=0.34.0 bash deploy/setup_gpu_box.sh   # builds both environments, runs the self-test
source ~/venvs/pagedserve/bin/activate
```

The `Makefile` looks for `.venv/bin/python`; with the environment active, pass `PY=python` to `make`.

If the self-test fails, **stop and release the instance**: a GPU that returns different results
for identical inputs makes every number meaningless.

Weights (about 2.5 GB, licence already accepted on your account) go in
`models/Llama-3.2-1B-Instruct/`. The quickest route is `scp` from the laptop, where they already
are: `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`.

## 2. Record the environment, once per environment

```bash
python -m bench.env_capture > results/env-engine.json                    # engine and HF baseline
~/venvs/vllm/bin/python -m bench.env_capture > results/env-vllm.json     # vLLM (records its version)
```

`bench.sweep --system vllm` refuses to start if its environment file does not record a vLLM
version.

## 3. Smoke test (first hour; correctness before any measurement)

```bash
python -m bench.gpu_selftest                         # again, as part of the record
PY=python make test                                  # CPU tests also pass here
PY=python make test-gpu                              # real weights on the GPU
python -m engine.server --model-dir models/Llama-3.2-1B-Instruct --dtype bfloat16 \
    --max-batch-size 32 --kv-cache-gb 8 --port 8000 &
python -m bench.loadgen run --url http://localhost:8000 --model llama-3.2-1b-instruct \
    --rate 2 --duration 20 --fetch-server-info --server-env results/env-engine.json \
    --out /tmp/smoke.jsonl                           # every request should succeed
python -m bench.loadgen summarize /tmp/smoke.jsonl
kill %1
python -m bench.greedy_eval --model-dir models/Llama-3.2-1B-Instruct --out results \
    --label <hardware>-bf16 --device cuda --dtype bfloat16 --prompts 50
```

Things most likely to go wrong, because they have never run on CUDA: the boolean attention mask in
batched decode (check it does not fall back to a very slow kernel), the indexed writes into the
KV pool, bf16 ties, and the `--kv-cache-gb` size. Note that `bench.greedy_eval` tests the plain
engine, not the scheduler path; there is not yet a GPU check of the scheduler against Hugging Face.

## 4. Measurement

Use fresh output directories (result files are never overwritten) and repeat with other seeds,
because the run-to-run spread was 10 to 19% (D-015). The tuned baseline from D-015 is batch size 32
with a 20 ms window.

```bash
# Hugging Face static batching, the tuned configuration, four seeds
for seed in 0 1 2 3; do
  python -m bench.sweep --system hf-static --model models/Llama-3.2-1B-Instruct \
      --out results/hf-static-<hw>-seed$seed --server-env results/env-engine.json \
      --batch-sizes 32 --windows-ms 20 --rates 2,4,8 --seed $seed
done

# This engine
for seed in 0 1 2 3; do
  python -m bench.sweep --system engine --model models/Llama-3.2-1B-Instruct \
      --out results/engine-<hw>-seed$seed --server-env results/env-engine.json \
      --batch-sizes 32 --rates 2,4,8 --kv-cache-gb 8 --seed $seed
done

# vLLM: matching features, then its defaults. Check `vllm serve --help` first, since the flag
# names are from memory of earlier versions and not verified for 0.34.0.
for seed in 0 1 2 3; do
  python -m bench.sweep --system vllm --vllm-python ~/venvs/vllm/bin/python \
      --model models/Llama-3.2-1B-Instruct --out results/vllm-<hw>-seed$seed \
      --server-env results/env-vllm.json --batch-sizes 32 --rates 2,4,8 --seed $seed
done
```

Prefix caching is on by default in this engine and in vLLM. For a comparison with it off, pass
`--server-arg=--no-prefix-caching` to the engine sweep and `--server-arg=--no-enable-prefix-caching`
to the vLLM sweep (the equals sign is required). State which was used.

The load generator's prompts are random token ids, so they share almost no prefixes: expect a
prefix-cache hit rate near zero in these runs (`/info` reports `prefix_hit_rate`). That is not a
failure, and it means these sweeps say nothing about the benefit of prefix caching. Showing that
needs a workload with shared prompt prefixes, which the load generator does not have yet.

Each sweep records a manifest with the system, extra server arguments, GPU load before and after,
and whether another user contended the GPU. On a dedicated instance nothing should be contended;
if a run is flagged, say so in the write-up.

## 5. Reports

```bash
python -m bench.sweep_report --dir results/engine-<hw>-seed0 --out results/engine-<hw>-seed0/REPORT.md
python -m bench.sweep_variation results/engine-<hw>-seed{0,1,2,3} --out results/engine-variation.md
```

Compare systems on the same rate, with the spread across seeds. Report where this engine is
slower, not only where it is faster.

## 6. Finishing (this is when the meter stops)

1. Copy `results/` back to the laptop (`scp -r`), check the files are complete, then commit them
   from the laptop. Look for hostnames, IPs and instance ids before committing.
2. **Stop or terminate the instance, and delete any attached storage**, in the provider's
   console. Check the console afterwards: stopped instances can still bill for disk.
3. Fill in D-018 and the measured effects in `docs/decisions.md`: what was measured, on what, and
   with which caveats.
