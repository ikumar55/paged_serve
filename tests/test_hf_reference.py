"""First CPU tests: our config maps faithfully onto a Hugging Face reference model.

These establish the harness that the forward-pass correctness tests will plug into.
"""

import pytest
import torch

from engine.config import LlamaConfig
from tests.hf_reference import build_reference_model

VARIANTS = {
    "plain": LlamaConfig.tiny(),
    "tied": LlamaConfig.tiny(tie_word_embeddings=True),
    "llama3_rope": LlamaConfig.tiny(llama3_rope_scaling=True),
    "tied_llama3_rope": LlamaConfig.tiny(tie_word_embeddings=True, llama3_rope_scaling=True),
}


@pytest.mark.parametrize("cfg", VARIANTS.values(), ids=VARIANTS.keys())
def test_parameter_count_matches_reference(cfg: LlamaConfig) -> None:
    model = build_reference_model(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == cfg.num_parameters()


@pytest.mark.parametrize("cfg", VARIANTS.values(), ids=VARIANTS.keys())
def test_reference_forward_is_deterministic_and_shaped(cfg: LlamaConfig) -> None:
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12), generator=torch.manual_seed(1))
    first = build_reference_model(cfg, seed=7)
    second = build_reference_model(cfg, seed=7)
    with torch.no_grad():
        logits_a = first(input_ids).logits
        logits_b = second(input_ids).logits
    assert logits_a.shape == (2, 12, cfg.vocab_size)
    assert logits_a.dtype == torch.float32
    assert torch.equal(logits_a, logits_b)


def test_llama3_rope_scaling_changes_logits() -> None:
    """Guards against the silent failure where rope scaling is dropped on the way to the model."""
    plain = LlamaConfig.tiny()
    scaled = LlamaConfig.tiny(llama3_rope_scaling=True)
    input_ids = torch.randint(0, plain.vocab_size, (1, 64), generator=torch.manual_seed(2))
    with torch.no_grad():
        out_plain = build_reference_model(plain, seed=3)(input_ids).logits
        out_scaled = build_reference_model(scaled, seed=3)(input_ids).logits
    assert not torch.allclose(out_plain, out_scaled)
