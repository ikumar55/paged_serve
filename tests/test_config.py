import dataclasses

import pytest

from engine.config import LlamaConfig, RopeScaling


def test_llama_3_2_1b_kv_sizing_matches_design_math() -> None:
    cfg = LlamaConfig.llama_3_2_1b()
    per_token = cfg.kv_bytes_per_token(dtype_bytes=2)
    assert per_token == 2 * 16 * 8 * 64 * 2 == 32 * 1024
    block_tokens = 16
    assert per_token * block_tokens == 512 * 1024  # one 16-token block
    assert 10 * 1024**3 // per_token == 327_680  # a 10 GiB pool, "roughly 300K tokens"


def test_head_dim_is_derived() -> None:
    assert LlamaConfig.tiny().head_dim == 16
    assert LlamaConfig.llama_3_2_1b().head_dim == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_heads": 3},  # hidden_size not divisible by num_heads
        {"num_kv_heads": 3},  # num_heads not divisible by num_kv_heads
        {"num_layers": 0},
        {"hidden_size": 60, "num_heads": 4},  # odd head_dim (15)
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(LlamaConfig.tiny(), **overrides)


def test_invalid_rope_scaling_is_rejected() -> None:
    with pytest.raises(ValueError):
        RopeScaling(
            factor=8.0,
            low_freq_factor=4.0,
            high_freq_factor=1.0,
            original_max_position_embeddings=8192,
        )


HF_STYLE_JSON = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 4.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
    "tie_word_embeddings": True,
}


def test_from_hf_dict_old_layout() -> None:
    cfg = LlamaConfig.from_hf_dict(HF_STYLE_JSON)
    assert cfg == LlamaConfig.tiny(tie_word_embeddings=True, llama3_rope_scaling=True)


def test_from_hf_dict_new_rope_parameters_layout() -> None:
    raw = {k: v for k, v in HF_STYLE_JSON.items() if k not in ("rope_theta", "rope_scaling")}
    raw["rope_parameters"] = {"rope_theta": 10000.0, **HF_STYLE_JSON["rope_scaling"]}  # type: ignore[dict-item]
    cfg = LlamaConfig.from_hf_dict(raw)
    assert cfg == LlamaConfig.tiny(tie_word_embeddings=True, llama3_rope_scaling=True)


def test_from_hf_dict_defaults_to_mha_and_plain_rope() -> None:
    raw = {
        k: v for k, v in HF_STYLE_JSON.items() if k not in ("num_key_value_heads", "rope_scaling")
    }
    cfg = LlamaConfig.from_hf_dict(raw)
    assert cfg.num_kv_heads == cfg.num_heads
    assert cfg.rope_scaling is None


def test_from_hf_dict_rejects_unsupported_rope_scaling() -> None:
    raw = dict(HF_STYLE_JSON, rope_scaling={"rope_type": "yarn", "factor": 2.0})
    with pytest.raises(ValueError, match="unsupported rope scaling"):
        LlamaConfig.from_hf_dict(raw)


def test_from_hf_dict_rejects_inconsistent_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        LlamaConfig.from_hf_dict(dict(HF_STYLE_JSON, head_dim=32))
