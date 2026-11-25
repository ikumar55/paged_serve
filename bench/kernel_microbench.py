"""Is the decode-attention kernel the lever? Isolated timing against the card's bandwidth (D-031).

For the shapes the engine runs (Llama-3.2-1B: 32 query heads, 8 key-value heads, head size 64,
block 16, 16 layers) this times one layer's decode attention for several implementations on
uniform contexts and on the real length mix of the saturated server (the contexts of D-020's
profile, batch 60), and reports achieved GB/s against two rooflines: the card's specified bandwidth
and a device-to-device copy measured in the same session. Every implementation first passes a
correctness gate against the float64 reference; one that fails is reported as failed and is not
timed. Timing uses a CUDA graph of one launch replayed after a 256 MB write has flushed the L2
cache (the cold figure) and a graph of 50 launches without flushing (the warm figure), so launch
overhead is not what is measured. The plan, predictions and decision rules are in
``docs/decisions.md`` (D-031 and its amendment 1).

    python -m bench.kernel_microbench --device cuda --out results/kernel-microbench-rtx5090.json \
        --step-profile results/d031-step-rtx5090.json

On a CPU (``--device cpu``) only the implementations that run there are timed; that mode exists
for the tests, not for measurements.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from bench.env_capture import capture_environment
from engine.model import attend
from engine.paged_attention import (
    decode_attention_reference,
    split_decode_attention_reference,
)

SPEC_BANDWIDTH_GBS = 1792.0  # the RTX 5090's specified memory bandwidth
UNIFORM_BATCHES = (1, 8, 32, 64)
UNIFORM_CONTEXTS = (128, 512, 1024, 2048, 4096)
MIX_SEEDS = (0, 1, 2, 3)
MIX_BATCH = 60
GATE_LENGTHS = (0, 15, 16, 31, 1, 40, 7, 200, 443, 1048)  # cached tokens, as in the GPU tests
FP32_GATE_LIMIT = 1e-5
BF16_GATE_FACTOR = 2.0  # of the baseline implementation's own error
LEVER_SHARE_LIMIT = 0.15  # at most this share of the step: attention is not the lever
ADOPTION_SPEEDUP = 1.15
CLOSE_TO_ROOFLINE = 0.60
BASELINE = "triton-w2"  # the engine's current kernel


@dataclass(frozen=True)
class Shape:
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 64
    block_size: int = 16
    layers: int = 16


@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # "uniform" or "mix"
    lengths: tuple[int, ...]  # cached tokens before the new one, per sequence

    @property
    def batch(self) -> int:
        return len(self.lengths)


def uniform_case(batch: int, context: int) -> Case:
    """``batch`` sequences that each attend over ``context`` positions (``context - 1`` cached)."""
    return Case(f"uniform-b{batch}-c{context}", "uniform", (context - 1,) * batch)


def mix_cases(seeds: Sequence[int], batch: int = MIX_BATCH) -> list[Case]:
    """The contexts of a saturated server's decode batch, drawn as D-020's profile draws them."""
    from bench.profile_decode import sample_mix_lengths, standard_requests

    return [
        Case(f"mix-s{seed}", "mix", tuple(sample_mix_lengths(standard_requests(seed), batch, seed)))
        for seed in seeds
    ]


def minimum_bytes(lengths: Sequence[int], shape: Shape, itemsize: int) -> int:
    """What a decode step must read at the least: every position's key and value of every head,
    plus the query and the output."""
    kv = sum(n + 1 for n in lengths) * 2 * shape.kv_heads * shape.head_dim * itemsize
    qo = 2 * len(lengths) * shape.heads * shape.head_dim * itemsize
    return kv + qo


# -- inputs ----------------------------------------------------------------------------------------


@dataclass
class Inputs:
    q: Tensor  # [batch, heads, 1, head_dim]
    k_pool: Tensor  # [num_blocks, kv_heads, block_size, head_dim]
    v_pool: Tensor
    tables: Tensor  # [batch, width] int64, padded with block 0
    lengths: Tensor  # [batch] int64
    shape: Shape


def build_inputs(
    case_lengths: Sequence[int], shape: Shape, dtype: torch.dtype, device: str, seed: int = 0
) -> Inputs:
    """A random pool with scrambled block tables, the way the allocator leaves them."""
    generator = torch.Generator().manual_seed(seed)
    need = [math.ceil((n + 1) / shape.block_size) for n in case_lengths]
    num_blocks = sum(need) + 1
    order = torch.randperm(num_blocks, generator=generator).tolist()
    tables = torch.zeros(len(case_lengths), max(need), dtype=torch.long)
    taken = 0
    for row, count in enumerate(need):
        tables[row, :count] = torch.tensor(order[taken : taken + count])
        taken += count

    def randn(*dims: int) -> Tensor:
        return torch.randn(*dims, generator=generator).to(dtype).to(device)

    pool_dims = (num_blocks, shape.kv_heads, shape.block_size, shape.head_dim)
    return Inputs(
        q=randn(len(case_lengths), shape.heads, 1, shape.head_dim),
        k_pool=randn(*pool_dims),
        v_pool=randn(*pool_dims),
        tables=tables.to(device),
        lengths=torch.tensor(list(case_lengths), dtype=torch.long, device=device),
        shape=shape,
    )


# -- implementations -------------------------------------------------------------------------------

Runner = Callable[
    [], Tensor
]  # one layer of decode attention on fixed inputs; [batch, heads, 1, dim]


@dataclass(frozen=True)
class Implementation:
    name: str
    description: str
    build: Callable[[Inputs], Runner]
    unavailable: Callable[[str], str | None]  # device -> why it cannot run (None: it can)


def _needs_triton(device: str) -> str | None:
    if device != "cuda" or not torch.cuda.is_available():
        return "needs a CUDA GPU"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "Triton is not installed"
    return None


def _gather(pool: Tensor, tables: Tensor) -> Tensor:
    batch, width = tables.shape
    blocks = pool[tables]  # [batch, width, kv_heads, block_size, head_dim]
    _, _, kv_heads, block_size, head_dim = blocks.shape
    return blocks.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, width * block_size, head_dim)


def _mask(inputs: Inputs) -> Tensor:
    padded = inputs.tables.shape[1] * inputs.shape.block_size
    columns = torch.arange(padded, device=inputs.q.device)
    return (columns.unsqueeze(0) <= inputs.lengths.unsqueeze(1)).view(-1, 1, 1, padded)


def build_gather(inputs: Inputs) -> Runner:
    """The path the kernel replaced: gather into padded tensors, mask, PyTorch attention."""
    mask = _mask(inputs)

    def run() -> Tensor:
        keys, values = _gather(inputs.k_pool, inputs.tables), _gather(inputs.v_pool, inputs.tables)
        return attend(inputs.q, keys, values, past=0, mask=mask)

    return run


def build_sdpa_contiguous(inputs: Inputs) -> Runner:
    """PyTorch attention on keys and values gathered beforehand (not timed): no gather cost."""
    keys, values = _gather(inputs.k_pool, inputs.tables), _gather(inputs.v_pool, inputs.tables)
    mask = _mask(inputs)
    return lambda: attend(inputs.q, keys, values, past=0, mask=mask)


def build_reference(inputs: Inputs) -> Runner:
    return lambda: decode_attention_reference(
        inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths
    )


def build_split_reference(splits: int) -> Callable[[Inputs], Runner]:
    def build(inputs: Inputs) -> Runner:
        return lambda: split_decode_attention_reference(
            inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths, splits
        )

    return build


def build_triton(warps: int) -> Callable[[Inputs], Runner]:
    def build(inputs: Inputs) -> Runner:
        from engine.triton_paged_attention import paged_decode_attention

        return lambda: paged_decode_attention(
            inputs.q, inputs.k_pool, inputs.v_pool, inputs.tables, inputs.lengths, num_warps=warps
        )

    return build


def build_triton_split(splits: int, warps: int) -> Callable[[Inputs], Runner]:
    def build(inputs: Inputs) -> Runner:
        from engine.triton_split_attention import paged_split_decode_attention

        return lambda: paged_split_decode_attention(
            inputs.q,
            inputs.k_pool,
            inputs.v_pool,
            inputs.tables,
            inputs.lengths,
            splits=splits,
            num_warps=warps,
        )

    return build


def _flash_attn_module() -> Any:
    try:
        from vllm import vllm_flash_attn  # the copy bundled with vLLM supports a block table

        return vllm_flash_attn
    except ImportError:
        import flash_attn

        return flash_attn


def _flash_attn_unavailable(device: str) -> str | None:
    if device != "cuda" or not torch.cuda.is_available():
        return "needs a CUDA GPU"
    try:
        _flash_attn_module()
    except ImportError:
        return "neither vllm's bundled flash_attn nor flash-attn is installed in this environment"
    return None


def build_flash_attn(inputs: Inputs) -> Runner:
    """FlashAttention's ``flash_attn_with_kvcache`` with a block table (written without a GPU)."""
    function = _flash_attn_module().flash_attn_with_kvcache
    k_cache = inputs.k_pool.permute(0, 2, 1, 3).contiguous()  # [blocks, block_size, kv_heads, dim]
    v_cache = inputs.v_pool.permute(0, 2, 1, 3).contiguous()
    q = inputs.q.permute(0, 2, 1, 3).contiguous()  # [batch, 1, heads, dim]
    lengths = (inputs.lengths + 1).to(torch.int32)
    tables = inputs.tables.to(torch.int32)

    def run() -> Tensor:
        out: Tensor = function(
            q, k_cache, v_cache, cache_seqlens=lengths, block_table=tables, causal=True
        )
        return out.permute(0, 2, 1, 3)

    return run


def _flashinfer_unavailable(device: str) -> str | None:
    if device != "cuda" or not torch.cuda.is_available():
        return "needs a CUDA GPU"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "flashinfer is not installed in this environment"
    return None


def build_flashinfer(inputs: Inputs) -> Runner:
    """FlashInfer's batch decode over a paged cache (written without a GPU; versions differ)."""
    import flashinfer

    shape = inputs.shape
    counts = (inputs.lengths + 1 + shape.block_size - 1) // shape.block_size
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
    indices = torch.cat([inputs.tables[row, : int(counts[row])] for row in range(len(counts))]).to(
        torch.int32
    )
    last_page_len = ((inputs.lengths % shape.block_size) + 1).to(torch.int32)
    k_cache = inputs.k_pool.permute(0, 2, 1, 3).contiguous()
    v_cache = inputs.v_pool.permute(0, 2, 1, 3).contiguous()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=inputs.q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        shape.heads,
        shape.kv_heads,
        shape.head_dim,
        shape.block_size,
        data_type=inputs.q.dtype,
        q_data_type=inputs.q.dtype,
    )
    q = inputs.q[:, :, 0, :].contiguous()
    return lambda: wrapper.run(q, (k_cache, v_cache)).unsqueeze(2)


def always(device: str) -> str | None:
    return None


def registry() -> dict[str, Implementation]:
    """Every implementation the tool knows, in the order they are reported."""
    items = [
        Implementation("gather-sdpa", "gather into padded tensors, then PyTorch attention",
                       build_gather, always),
        Implementation("sdpa-contiguous", "PyTorch attention on pre-gathered keys and values",
                       build_sdpa_contiguous, always),
        Implementation("reference", "float64 reference, a Python loop (CPU tests only)",
                       build_reference, always),
    ]  # fmt: skip
    for splits in (2, 3):
        items.append(
            Implementation(f"split-ref-p{splits}", "split-KV algorithm in float64 (CPU tests only)",
                           build_split_reference(splits), always)
        )  # fmt: skip
    for warps in (1, 2, 4, 8):
        items.append(
            Implementation(f"triton-w{warps}", f"engine's Triton kernel, {warps} warps",
                           build_triton(warps), _needs_triton)
        )  # fmt: skip
    for warps in (2, 4):
        for splits in (2, 4, 8):
            items.append(
                Implementation(f"triton-split{splits}-w{warps}",
                               f"split-KV Triton kernel, {splits} splits, {warps} warps",
                               build_triton_split(splits, warps), _needs_triton)
            )  # fmt: skip
    items.append(Implementation("flash-attn-paged", "FlashAttention decode over a block table",
                                build_flash_attn, _flash_attn_unavailable))  # fmt: skip
    items.append(Implementation("flashinfer", "FlashInfer batch decode, paged cache",
                                build_flashinfer, _flashinfer_unavailable))  # fmt: skip
    return {item.name: item for item in items}


DEFAULT_IMPLS = (
    "gather-sdpa",
    "sdpa-contiguous",
    "triton-w1",
    "triton-w2",
    "triton-w4",
    "triton-w8",
    "triton-split2-w2",
    "triton-split4-w2",
    "triton-split8-w2",
    "triton-split2-w4",
    "triton-split4-w4",
    "triton-split8-w4",
    "flash-attn-paged",
    "flashinfer",
)


# -- the correctness gate --------------------------------------------------------------------------


@dataclass
class Gate:
    passed: bool
    fp32_error: float | None = None
    bf16_error: float | None = None
    bf16_limit: float | None = None
    reason: str | None = None


def max_error(runner: Runner, truth: Tensor) -> float:
    out = runner()
    if out.shape != truth.shape:
        raise ValueError(f"output shape {tuple(out.shape)}, expected {tuple(truth.shape)}")
    return float((out.double().cpu() - truth.cpu()).abs().max())


def run_gate(impl: Implementation, baseline: Implementation, shape: Shape, device: str) -> Gate:
    """fp32 error at most 1e-5 against the float64 reference; bf16 at most twice the baseline's.

    Any exception (a different API, a compile error, a wrong shape) fails the gate with its text.
    """
    try:
        fp32 = build_inputs(GATE_LENGTHS, shape, torch.float32, device)
        truth32 = decode_attention_reference(
            fp32.q, fp32.k_pool, fp32.v_pool, fp32.tables, fp32.lengths, dtype=torch.float64
        )
        fp32_error = max_error(impl.build(fp32), truth32)
        bf16 = build_inputs(GATE_LENGTHS, shape, torch.bfloat16, device, seed=1)
        truth16 = decode_attention_reference(
            bf16.q, bf16.k_pool, bf16.v_pool, bf16.tables, bf16.lengths, dtype=torch.float64
        )
        bf16_error = max_error(impl.build(bf16), truth16)
        base_error = max_error(baseline.build(bf16), truth16)
    except Exception as error:  # a failing variant is a result, not a crash
        return Gate(False, reason=f"{type(error).__name__}: {error}"[:300])
    limit = BF16_GATE_FACTOR * max(base_error, 1e-6)
    passed = fp32_error <= FP32_GATE_LIMIT and bf16_error <= limit
    reason = None if passed else "error above the gate"
    return Gate(passed, fp32_error, bf16_error, limit, reason)


# -- timing ----------------------------------------------------------------------------------------


@dataclass
class Timing:
    mode: str  # "graph" or "eager"
    cold_s: list[float] = field(default_factory=list)  # one per repetition, L2 flushed before
    warm_s: float = 0.0  # one launch, median over repetitions of a 50-launch graph, no flushing


class Timer(Protocol):
    def measure(self, run: Runner, repeats: int, warmup: int) -> Timing: ...


WARM_LAUNCHES = 50
FLUSH_BYTES = 256 * 1024 * 1024  # larger than the RTX 5090's 96 MB L2


class CudaTimer:
    """CUDA events around the replay of a captured graph; falls back to eager launches."""

    def __init__(self) -> None:
        self.flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")

    def _capture(self, run: Runner, launches: int) -> torch.cuda.CUDAGraph | None:
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(launches):
                    run()
        except RuntimeError:
            return None
        return graph

    def _events(self, launch: Callable[[], None]) -> float:
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) / 1000.0

    def measure(self, run: Runner, repeats: int, warmup: int) -> Timing:
        single = self._capture(run, 1)
        many = self._capture(run, WARM_LAUNCHES)
        mode = "graph" if single is not None and many is not None else "eager"

        def launch_eager() -> None:
            run()

        launch_one: Callable[[], None] = single.replay if single is not None else launch_eager
        timing = Timing(mode)
        for index in range(warmup + repeats):
            self.flush.zero_()  # push the pools out of L2
            seconds = self._events(launch_one)
            if index >= warmup:
                timing.cold_s.append(seconds)
        if many is not None:
            warm = [self._events(many.replay) / WARM_LAUNCHES for _ in range(max(5, repeats // 10))]
        else:
            warm = [self._events(launch_eager) for _ in range(max(5, repeats // 10))]
        timing.warm_s = statistics.median(warm)
        return timing


class CpuTimer:
    """Wall-clock timing without graphs or cache flushing: for the tests, not for measurements."""

    def measure(self, run: Runner, repeats: int, warmup: int) -> Timing:
        timing = Timing("eager")
        for index in range(warmup + repeats):
            start = time.perf_counter()
            run()
            if index >= warmup:
                timing.cold_s.append(time.perf_counter() - start)
        timing.warm_s = statistics.median(timing.cold_s)
        return timing


def copy_gbs(size_bytes: int, seconds: float) -> float:
    """A copy reads and writes the buffer once each."""
    return 2 * size_bytes / seconds / 1e9


def copy_bandwidth_gbs(device: str, megabytes: int = 1024, repeats: int = 20) -> float:
    """The best device-to-device copy bandwidth in GB/s (bytes read plus bytes written)."""
    size = megabytes * 1024 * 1024
    source = torch.zeros(size, dtype=torch.uint8, device=device)
    target = torch.empty_like(source)
    best = 0.0
    for index in range(repeats + 3):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            target.copy_(source)
            end.record()
            torch.cuda.synchronize()
            seconds = float(start.elapsed_time(end)) / 1000.0
        else:
            begin = time.perf_counter()
            target.copy_(source)
            seconds = time.perf_counter() - begin
        if index >= 3:
            best = max(best, copy_gbs(size, seconds))
    return best


def percentiles_us(values: Sequence[float]) -> tuple[float, float, float]:
    p10, p50, p90 = np.percentile(np.asarray(values, dtype=float) * 1e6, [10, 50, 90])
    return float(p10), float(p50), float(p90)


# -- the benchmark ---------------------------------------------------------------------------------


@dataclass
class Config:
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    impls: tuple[str, ...] = DEFAULT_IMPLS
    uniform_batches: tuple[int, ...] = UNIFORM_BATCHES
    uniform_contexts: tuple[int, ...] = UNIFORM_CONTEXTS
    mix_seeds: tuple[int, ...] = MIX_SEEDS
    mix_batch: int = MIX_BATCH
    repeats: int = 200
    warmup: int = 20
    shape: Shape = field(default_factory=Shape)
    copy_megabytes: int = 1024


def make_cases(config: Config) -> list[Case]:
    cases = [uniform_case(b, c) for b in config.uniform_batches for c in config.uniform_contexts]
    return cases + (mix_cases(config.mix_seeds, config.mix_batch) if config.mix_seeds else [])


def time_case(
    case: Case, impl: Implementation, config: Config, timer: Timer, copy_gbs: float
) -> dict[str, Any]:
    inputs = build_inputs(case.lengths, config.shape, config.dtype, config.device)
    timing = timer.measure(impl.build(inputs), config.repeats, config.warmup)
    p10, p50, p90 = percentiles_us(timing.cold_s)
    itemsize = torch.tensor([], dtype=config.dtype).element_size()
    needed = minimum_bytes(case.lengths, config.shape, itemsize)
    cold_gbs = needed / (p50 * 1e-6) / 1e9
    return {
        "case": case.name,
        "kind": case.kind,
        "batch": case.batch,
        "positions": sum(n + 1 for n in case.lengths),
        "minimum_bytes": needed,
        "impl": impl.name,
        "mode": timing.mode,
        "cold_p10_us": p10,
        "cold_p50_us": p50,
        "cold_p90_us": p90,
        "warm_p50_us": timing.warm_s * 1e6,
        "cold_gbs": cold_gbs,
        "fraction_of_copy": cold_gbs / copy_gbs,
        "fraction_of_spec": cold_gbs / SPEC_BANDWIDTH_GBS,
    }


def run_benchmark(
    config: Config, timer: Timer, log: Callable[[str], None] = lambda text: None
) -> dict[str, Any]:
    """The whole measurement: roofline, gates, then every case of every implementation passing."""
    known = registry()
    unknown = [name for name in config.impls if name not in known]
    if unknown:
        raise ValueError(f"unknown implementations: {unknown}")
    baseline = known["gather-sdpa"] if BASELINE not in config.impls else known[BASELINE]
    if baseline.unavailable(config.device):
        baseline = known["gather-sdpa"]
    copy_gbs = copy_bandwidth_gbs(config.device, config.copy_megabytes)
    log(f"copy bandwidth {copy_gbs:.0f} GB/s (specified {SPEC_BANDWIDTH_GBS:.0f})")

    unavailable: dict[str, str] = {}
    gates: dict[str, dict[str, Any]] = {}
    passing: list[Implementation] = []
    for name in config.impls:
        impl = known[name]
        reason = impl.unavailable(config.device)
        if reason is not None:
            unavailable[name] = reason
            log(f"{name}: not available ({reason})")
            continue
        gate = run_gate(impl, baseline, config.shape, config.device)
        gates[name] = vars(gate)
        log(f"{name}: gate {'passed' if gate.passed else 'FAILED: ' + str(gate.reason)}")
        if gate.passed:
            passing.append(impl)

    results = []
    for case in make_cases(config):
        for impl in passing:
            try:
                row = time_case(case, impl, config, timer, copy_gbs)
            except Exception as error:  # recorded, not fatal
                gates[impl.name]["passed"] = False
                gates[impl.name]["reason"] = f"failed while timing {case.name}: {error}"[:300]
                continue
            results.append(row)
            log(f"{case.name} {impl.name}: {row['cold_p50_us']:.1f} us, {row['cold_gbs']:.0f} GB/s")
    return {
        "device": config.device,
        "dtype": str(config.dtype).removeprefix("torch."),
        "shape": vars(config.shape),
        "copy_bandwidth_gbs": copy_gbs,
        "spec_bandwidth_gbs": SPEC_BANDWIDTH_GBS,
        "baseline": baseline.name,
        "repeats": config.repeats,
        "warmup": config.warmup,
        "gates": gates,
        "unavailable": unavailable,
        "results": results,
    }


# -- the analysis: the lever test and the predictions -------------------------------------


def mix_times(doc: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per implementation, the cold median in microseconds on each mix case (impl, then case)."""
    table: dict[str, dict[str, float]] = {}
    for row in doc["results"]:
        if row["kind"] == "mix":
            table.setdefault(row["impl"], {})[row["case"]] = row["cold_p50_us"]
    return table


def read_step_profile(path: Path) -> dict[str, float]:
    """The median decode step with graphs, in milliseconds, per mix case (``mix-s<seed>``)."""
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"mix-s{entry['workload_seed']}": float(entry["wall_ms_per_step_median"])
        for entry in record["mix"]
        if entry["attention"].endswith("+graphs")
    }


def lever_test(
    doc: dict[str, Any], step_ms: dict[str, float], layers: int = Shape().layers
) -> dict[str, Any]:
    """Attention's share of the decode step and what the best variant would give (D-031).

    share = layers x (the current kernel's time for one layer) / (the step), averaged over the mix
    cases that have both. Even an infinitely fast kernel could shorten the step only by that share.
    """
    times = mix_times(doc)
    current = times.get(BASELINE)
    if not current:
        return {"verdict": "not evaluable: the current kernel was not timed on the mix"}
    shares = {
        case: layers * us * 1e-3 / step_ms[case] for case, us in current.items() if case in step_ms
    }
    if not shares:
        return {"verdict": "not evaluable: no decode-step time for the mix cases (--step-profile)"}
    share = statistics.fmean(shares.values())
    best_name, best_us = BASELINE, statistics.fmean(current.values())
    for name, per_case in times.items():
        if len(per_case) == len(current) and name.startswith("triton"):
            mean_us = statistics.fmean(per_case.values())
            if mean_us < best_us:
                best_name, best_us = name, mean_us
    speedup = statistics.fmean(current.values()) / best_us
    bound = 1.0 / (1.0 - share) if share < 1 else math.inf
    if share <= LEVER_SHARE_LIMIT:
        verdict = "attention is NOT the lever: even an infinitely fast kernel gives little"
    elif speedup >= ADOPTION_SPEEDUP:
        verdict = (
            f"candidate: {best_name} is {speedup:.2f} times faster than the current kernel; "
            "adopting it needs its own plan and the gates of D-025 and D-020 again"
        )
    else:
        verdict = (
            f"no variant tried is at least {ADOPTION_SPEEDUP} times faster than the current one"
        )
    return {
        "share_of_step": share,
        "step_speedup_bound": bound,
        "best_variant": best_name,
        "best_speedup_over_current": speedup,
        "verdict": verdict,
    }


Row = tuple[str, str, str, str]


def _row(name: str, written: str, value: float | None, low: float, high: float, fmt: str) -> Row:
    if value is None:
        return name, written, "not evaluable", "not evaluable"
    return name, written, format(value, fmt), "held" if low <= value <= high else "missed"


def prediction_rows(doc: dict[str, Any], step_ms: dict[str, float]) -> list[Row]:
    """The predictions of D-031 (as amended), each held or missed."""
    times = mix_times(doc)
    cur = times.get(BASELINE, {})
    cur_mean = statistics.fmean(cur.values()) if cur else None

    def mean_of(name: str) -> float | None:
        values = times.get(name)
        return statistics.fmean(values.values()) if values else None

    mix_rows = [r for r in doc["results"] if r["kind"] == "mix" and r["impl"] == BASELINE]
    fraction = statistics.fmean(r["fraction_of_copy"] for r in mix_rows) if mix_rows else None
    gather = mean_of("gather-sdpa")
    splits = [n for n in times if n.startswith("triton-split")]
    best_split = min((mean_of(n) for n in splits), default=None, key=lambda v: v or math.inf)
    warps = {n: mean_of(n) for n in ("triton-w1", "triton-w2", "triton-w4", "triton-w8")}
    warp_times = [v for v in warps.values() if v is not None]
    lever = lever_test(doc, step_ms)
    rows = [
        _row(
            "(1') current kernel, mix, microseconds per layer",
            "200 to 320",
            cur_mean,
            200,
            320,
            ".0f",
        ),
        _row(
            "(1') current kernel, share of measured copy bandwidth",
            "0.08 to 0.20",
            fraction,
            0.08,
            0.20,
            ".3f",
        ),
        _row(
            "(2) gather path over current kernel, mix",
            "at least 3",
            None if not (gather and cur_mean) else gather / cur_mean,
            3,
            math.inf,
            ".2f",
        ),
        _row(
            "(3') best split-KV over current kernel, mix",
            "at least 1.3",
            None if not (best_split and cur_mean) else cur_mean / best_split,
            1.3,
            math.inf,
            ".2f",
        ),
        _row(
            "(4) current kernel over best warp count, mix",
            "at most 1.15",
            None
            if not (warp_times and warps["triton-w2"])
            else warps["triton-w2"] / min(warp_times),
            0,
            1.15,
            ".2f",
        ),
        _row(
            "(6') attention's share of the decode step",
            "0.45 to 0.60",
            lever.get("share_of_step"),
            0.45,
            0.60,
            ".2f",
        ),
    ]
    flash = mean_of("flash-attn-paged")
    rows.append(
        _row(
            "(5) current kernel over FlashAttention, mix",
            "1.5 to 4",
            None if not (flash and cur_mean) else cur_mean / flash,
            1.5,
            4,
            ".2f",
        )
    )
    return rows


def render_report(doc: dict[str, Any], step_ms: dict[str, float]) -> str:
    lines = [
        "# D-031: the decode-attention kernel in isolation",
        "",
        f"Device {doc['device']}, {doc['dtype']}, one layer of decode attention. Copy bandwidth "
        f"measured in this session: {doc['copy_bandwidth_gbs']:.0f} GB/s (specified "
        f"{doc['spec_bandwidth_gbs']:.0f}). GB/s is the minimum bytes a step must read over the "
        "cold median time. Figures are medians of cold-cache repetitions (L2 flushed first).",
        "",
        "## Implementations",
        "",
        "| name | gate | fp32 error | bf16 error | bf16 limit | note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    def sci(value: float | None) -> str:
        return "-" if value is None else f"{value:.2e}"

    for name, gate in doc["gates"].items():
        lines.append(
            f"| {name} | {'passed' if gate['passed'] else 'FAILED'} | {sci(gate['fp32_error'])} "
            f"| {sci(gate['bf16_error'])} | {sci(gate['bf16_limit'])} | {gate['reason'] or ''} |"
        )
    for name, reason in doc["unavailable"].items():
        lines.append(f"| {name} | not available | - | - | - | {reason} |")

    for kind, title in (("uniform", "Uniform contexts"), ("mix", "The real length mix (batch 60)")):
        rows = [r for r in doc["results"] if r["kind"] == kind]
        if not rows:
            continue
        lines += [
            "",
            f"## {title}",
            "",
            "| case | implementation | cold p50 us | p10 to p90 us | warm p50 us | GB/s "
            "| of copy | of spec |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            lines.append(
                f"| {r['case']} | {r['impl']} | {r['cold_p50_us']:.1f} | {r['cold_p10_us']:.1f} to "
                f"{r['cold_p90_us']:.1f} | {r['warm_p50_us']:.1f} | {r['cold_gbs']:.0f} "
                f"| {r['fraction_of_copy']:.2f} | {r['fraction_of_spec']:.2f} |"
            )
    lever = lever_test(doc, step_ms)
    lines += ["", "## The lever test", ""]
    for key in ("share_of_step", "step_speedup_bound", "best_variant", "best_speedup_over_current"):
        if key in lever:
            value = lever[key]
            lines.append(
                f"- {key}: {value:.3f}" if isinstance(value, float) else f"- {key}: {value}"
            )
    lines += [f"- **{lever['verdict']}**", ""]
    lines += [
        "## Predictions of D-031 (amendment 1), each held or missed",
        "",
        "| prediction | written | measured | verdict |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {a} | {b} | {c} | **{d}** |" for a, b, c, d in prediction_rows(doc, step_ms)]
    best = max((r["fraction_of_copy"] for r in doc["results"]), default=0.0)
    if best > CLOSE_TO_ROOFLINE:
        lines += [
            "",
            f"An implementation reached {best:.0%} of the measured copy bandwidth: "
            "close to what the card allows at these shapes.",
        ]
    lines.append("")
    return "\n".join(lines)


# -- command line ----------------------------------------------------------------------------------


def merge_documents(main: dict[str, Any], extra: dict[str, Any], label: str) -> dict[str, Any]:
    """``main`` with the results, gates and unavailable entries of ``extra`` (another environment).

    FlashAttention and FlashInfer live in the vLLM environment, whose PyTorch is not the engine's.
    Their rows keep their own bandwidth fractions (computed from that run's copy roofline) and say
    which environment they came from; a name already in ``main`` is never overwritten.
    """
    merged: dict[str, Any] = json.loads(json.dumps(main))  # a deep copy
    for row in extra["results"]:
        if row["impl"] not in main["gates"] and row["impl"] not in main["unavailable"]:
            merged["results"].append({**row, "environment": label})
    for name, gate in extra["gates"].items():
        if name not in merged["gates"] and name not in merged["unavailable"]:
            merged["gates"][name] = {
                **gate,
                "reason": f"[{label}] {gate.get('reason') or ''}".strip(),
            }
    for name, reason in extra["unavailable"].items():
        if name not in merged["gates"] and name not in merged["unavailable"]:
            merged["unavailable"][name] = f"[{label}] {reason}"
    merged.setdefault("merged_from", []).append(
        {"label": label, "copy_bandwidth_gbs": extra["copy_bandwidth_gbs"]}
    )
    return merged


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("torch", "triton", "flash-attn", "flashinfer-python", "vllm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--out", type=Path, required=True, help="JSON result (never overwritten)")
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seeds", default=",".join(str(s) for s in MIX_SEEDS))
    parser.add_argument("--mix-batch", type=int, default=MIX_BATCH)
    parser.add_argument("--uniform-batches", default=",".join(str(b) for b in UNIFORM_BATCHES))
    parser.add_argument("--uniform-contexts", default=",".join(str(c) for c in UNIFORM_CONTEXTS))
    parser.add_argument("--copy-mb", type=int, default=1024, help="buffer of the copy roofline")
    parser.add_argument("--step-profile", type=Path, help="bench.profile_decode mix JSON (graphs)")
    parser.add_argument(
        "--report",
        type=Path,
        help="do not measure: render the report (to --out) of this earlier result JSON",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="*",
        default=[],
        help="with --report: result JSONs of other environments (the vLLM one) to include",
    )
    return parser


def report_only(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"error: {args.out} already exists; results are never overwritten", file=sys.stderr)
        return 2
    doc = json.loads(args.report.read_text(encoding="utf-8"))
    for path in args.merge:
        doc = merge_documents(doc, json.loads(path.read_text(encoding="utf-8")), path.stem)
    step_ms = read_step_profile(args.step_profile) if args.step_profile else {}
    args.out.write_text(render_report(doc, step_ms), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report is not None:
        return report_only(args)
    if args.out.exists():
        print(f"error: {args.out} already exists; results are never overwritten", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda but no CUDA GPU is available", file=sys.stderr)
        return 2
    if args.device == "cuda":  # "fp32" must mean fp32, as in the accuracy gates
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = Config(
        device=args.device,
        dtype=getattr(torch, args.dtype),
        impls=tuple(name for name in args.impls.split(",") if name),
        mix_seeds=tuple(int(s) for s in args.seeds.split(",") if s),
        mix_batch=args.mix_batch,
        uniform_batches=tuple(int(b) for b in args.uniform_batches.split(",") if b),
        uniform_contexts=tuple(int(c) for c in args.uniform_contexts.split(",") if c),
        copy_megabytes=args.copy_mb,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    timer: Timer = CudaTimer() if args.device == "cuda" else CpuTimer()
    doc = run_benchmark(config, timer, log=lambda text: print(text, file=sys.stderr, flush=True))
    environment = capture_environment(args=sys.argv[1:] if argv is None else list(argv))
    doc["environment"] = environment
    doc["package_versions"] = package_versions()
    step_ms = read_step_profile(args.step_profile) if args.step_profile else {}
    doc["lever_test"] = lever_test(doc, step_ms)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report = args.out.with_suffix(".md")
    report.write_text(render_report(doc, step_ms), encoding="utf-8")
    print(f"wrote {args.out} and {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
