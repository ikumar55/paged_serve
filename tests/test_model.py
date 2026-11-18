"""Correctness of the engine forward pass against the Hugging Face reference (fp32, CPU)."""

import pytest
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from engine.config import LlamaConfig
from engine.generate import greedy_generate
from engine.kv_cache import ContiguousKVCache
from engine.model import Llama, rope_inv_freq
from engine.weights import load_hf_state_dict
from tests.hf_reference import build_reference_model, to_hf_config

# The project target is 1e-3 max absolute logit difference in fp32 (docs/decisions.md, D-010).
# Differences here are far smaller, so the tighter bound catches regressions the target would hide.
FP32_LOGIT_TOL = 1e-4

VARIANTS = {
    "plain": LlamaConfig.tiny(),
    "tied": LlamaConfig.tiny(tie_word_embeddings=True),
    "llama3_rope": LlamaConfig.tiny(llama3_rope_scaling=True),
    "tied_llama3_rope": LlamaConfig.tiny(tie_word_embeddings=True, llama3_rope_scaling=True),
}
every_variant = pytest.mark.parametrize("cfg", VARIANTS.values(), ids=VARIANTS.keys())


def make_pair(cfg: LlamaConfig, seed: int = 0) -> tuple[LlamaForCausalLM, Llama]:
    reference = build_reference_model(cfg, seed)
    ours = Llama(cfg)
    load_hf_state_dict(ours, reference.state_dict())
    ours.eval()
    return reference, ours


def random_ids(cfg: LlamaConfig, batch: int, length: int, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (batch, length), generator=torch.manual_seed(seed))


@every_variant
def test_logits_match_reference(cfg: LlamaConfig) -> None:
    reference, ours = make_pair(cfg)
    ids = random_ids(cfg, batch=3, length=21)
    with torch.no_grad():
        expected = reference(ids).logits
        actual = ours(ids)
    assert expected.abs().max() > 0.5  # logits are not trivially small
    assert (expected - actual).abs().max() < FP32_LOGIT_TOL


@every_variant
def test_incremental_decoding_matches_full_forward(cfg: LlamaConfig) -> None:
    _, ours = make_pair(cfg)
    ids = random_ids(cfg, batch=2, length=19)
    prompt_len = 7
    with torch.no_grad():
        full = ours(ids)
        cache = ContiguousKVCache(cfg, 2, 32, ours.dtype, ours.device)
        pieces = [ours(ids[:, :prompt_len], cache=cache)]
        for position in range(prompt_len, ids.shape[1]):
            pieces.append(ours(ids[:, position : position + 1], cache=cache))
    assert cache.length == ids.shape[1]
    # Logits here reach about 18, and attention over different shapes accumulates in a different
    # order, so float32 rounding noise of a few 1e-5 is expected; a real bug would be far larger.
    assert (torch.cat(pieces, dim=1) - full).abs().max() < FP32_LOGIT_TOL


@every_variant
def test_chunked_prefill_matches_full_forward(cfg: LlamaConfig) -> None:
    _, ours = make_pair(cfg)
    ids = random_ids(cfg, batch=2, length=23)
    with torch.no_grad():
        full = ours(ids)
        cache = ContiguousKVCache(cfg, 2, 32, ours.dtype, ours.device)
        chunks = [ours(ids[:, start : start + 5], cache=cache) for start in range(0, 23, 5)]
    assert (torch.cat(chunks, dim=1) - full).abs().max() < FP32_LOGIT_TOL


def test_last_only_returns_final_position() -> None:
    cfg = LlamaConfig.tiny()
    _, ours = make_pair(cfg)
    ids = random_ids(cfg, batch=2, length=9)
    with torch.no_grad():
        full = ours(ids)
        last = ours(ids, last_only=True)
    assert last.shape == (2, 1, cfg.vocab_size)
    # Not bit-identical: a one-row matmul can use a different kernel than the full one.
    assert (last[:, 0] - full[:, -1]).abs().max() < FP32_LOGIT_TOL


def test_llama3_rope_frequencies_match_reference_at_real_scale() -> None:
    cfg = LlamaConfig.llama_3_2_1b()
    assert cfg.rope_scaling is not None
    reference = LlamaRotaryEmbedding(config=to_hf_config(cfg)).inv_freq
    ours = rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=0)


def test_llama3_rope_scaling_bands() -> None:
    cfg = LlamaConfig.llama_3_2_1b()
    assert cfg.rope_scaling is not None
    plain = rope_inv_freq(cfg.head_dim, cfg.rope_theta, None)
    scaled = rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
    ratio = scaled / plain
    assert ratio[0].item() == pytest.approx(1.0)  # highest frequency untouched
    assert ratio[-1].item() == pytest.approx(1 / cfg.rope_scaling.factor)  # lowest slowed fully
    assert torch.all(ratio[1:] <= ratio[:-1] + 1e-7)  # monotone in between
    assert ((ratio > 1 / cfg.rope_scaling.factor + 1e-3) & (ratio < 1 - 1e-3)).any()  # blended band


@every_variant
def test_greedy_generation_matches_reference(cfg: LlamaConfig) -> None:
    reference, ours = make_pair(cfg)
    eos = reference.generation_config.eos_token_id
    for seed, prompt_len in enumerate([1, 3, 8, 15, 30], start=10):
        prompt = random_ids(cfg, 1, prompt_len, seed=seed)
        # Pass an explicit attention mask: without one, generate() infers it from pad_token_id and
        # silently masks any prompt token that happens to equal that id (token 0 here).
        expected = reference.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=0,
        )[0, prompt_len:].tolist()
        actual = greedy_generate(ours, prompt[0].tolist(), 12, eos_token_id=eos)
        assert actual == expected


def test_generation_stops_at_eos() -> None:
    cfg = LlamaConfig.tiny()
    _, ours = make_pair(cfg)
    prompt = random_ids(cfg, 1, 6)[0].tolist()
    unrestricted = greedy_generate(ours, prompt, 8)
    assert len(unrestricted) == 8
    stopped = greedy_generate(ours, prompt, 8, eos_token_id=unrestricted[2])
    assert stopped == unrestricted[: unrestricted.index(unrestricted[2]) + 1]


def test_generation_rejects_bad_requests() -> None:
    cfg = LlamaConfig.tiny()
    _, ours = make_pair(cfg)
    with pytest.raises(ValueError, match="at least one token"):
        greedy_generate(ours, [], 4)
    with pytest.raises(ValueError, match="positive"):
        greedy_generate(ours, [1], 0)
    with pytest.raises(ValueError, match="maximum"):
        greedy_generate(ours, [1] * 250, 10)


def test_tied_checkpoint_without_lm_head_loads() -> None:
    cfg = LlamaConfig.tiny(tie_word_embeddings=True)
    reference = build_reference_model(cfg)
    state = {k: v for k, v in reference.state_dict().items() if k != "lm_head.weight"}
    ours = Llama(cfg)
    load_hf_state_dict(ours, state)
    ids = random_ids(cfg, 1, 5)
    with torch.no_grad():
        assert (reference(ids).logits - ours(ids)).abs().max() < FP32_LOGIT_TOL


def test_loader_rejects_bad_checkpoints() -> None:
    cfg = LlamaConfig.tiny()
    state = dict(build_reference_model(cfg).state_dict())

    missing = {k: v for k, v in state.items() if k != "model.norm.weight"}
    with pytest.raises(ValueError, match="missing"):
        load_hf_state_dict(Llama(cfg), missing)

    with pytest.raises(ValueError, match="unexpected"):
        load_hf_state_dict(Llama(cfg), {**state, "model.extra.weight": torch.zeros(1)})

    wrong_shape = {**state, "model.norm.weight": torch.zeros(cfg.hidden_size + 1)}
    with pytest.raises(ValueError, match="shape mismatch"):
        load_hf_state_dict(Llama(cfg), wrong_shape)


def test_loader_ignores_stored_rotary_frequencies() -> None:
    cfg = LlamaConfig.tiny()
    state = dict(build_reference_model(cfg).state_dict())
    state["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(cfg.head_dim // 2)
    load_hf_state_dict(Llama(cfg), state)


def test_kv_cache_rejects_overflow() -> None:
    cfg = LlamaConfig.tiny()
    cache = ContiguousKVCache(cfg, 1, 4, torch.float32, "cpu")
    k = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, k, k)


def test_bf16_model_runs_and_keeps_rope_frequencies_in_fp32() -> None:
    cfg = LlamaConfig.tiny(llama3_rope_scaling=True)
    reference, _ = make_pair(cfg)
    ours = Llama(cfg, dtype=torch.bfloat16)
    load_hf_state_dict(ours, reference.state_dict())
    ids = random_ids(cfg, 1, 10)
    with torch.no_grad():
        logits = ours(ids)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
    assert ours.rotary.inv_freq.dtype == torch.float32
