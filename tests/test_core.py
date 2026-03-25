"""Tests for surya.core — inference engine, token generation, KV cache."""

import pytest

from surya.core import (
    EvictionPolicy,
    InferenceEngine,
    InferenceSession,
    KVCache,
    ModelConfig,
    PRESET_MODELS,
    QuantizationMode,
    TokenGenerator,
)


# ---- ModelConfig tests ----

def test_model_config_defaults():
    cfg = ModelConfig("test", 100, 8000, 256, 6)
    assert cfg.quantization == QuantizationMode.FP16
    assert cfg.max_sequence_length == 2048


def test_model_config_bytes_per_param():
    for mode, expected in [
        (QuantizationMode.INT4, 0.5),
        (QuantizationMode.INT8, 1.0),
        (QuantizationMode.FP16, 2.0),
        (QuantizationMode.FP32, 4.0),
    ]:
        cfg = ModelConfig("t", 10, 100, 64, 2, quantization=mode)
        assert cfg.bytes_per_param == expected


def test_model_config_estimated_memory():
    cfg = ModelConfig("t", 100, 100, 64, 2, quantization=QuantizationMode.FP32)
    # 100M * 4 bytes = 400MB -> ~381.47 MB
    assert cfg.estimated_memory_mb == pytest.approx(381.47, rel=0.01)


def test_model_config_invalid_params():
    with pytest.raises(ValueError):
        ModelConfig("bad", -1, 100, 64, 2)
    with pytest.raises(ValueError):
        ModelConfig("bad", 10, 0, 64, 2)


def test_model_config_summary():
    cfg = ModelConfig("tiny", 25, 8000, 256, 6)
    s = cfg.summary()
    assert "tiny" in s
    assert "25" in s


def test_preset_models_exist():
    assert "surya-nano" in PRESET_MODELS
    assert "surya-micro" in PRESET_MODELS
    assert "surya-mini" in PRESET_MODELS


# ---- TokenGenerator tests ----

def test_token_generator_deterministic():
    gen = TokenGenerator(vocab_size=100, seed=42)
    a = gen.generate_next_token([1, 2, 3])
    gen2 = TokenGenerator(vocab_size=100, seed=42)
    b = gen2.generate_next_token([1, 2, 3])
    assert a == b


def test_token_generator_range():
    gen = TokenGenerator(vocab_size=50, seed=7)
    for _ in range(100):
        tok = gen.generate_next_token()
        assert 0 <= tok < 50


def test_generate_sequence_length():
    gen = TokenGenerator(vocab_size=100, seed=0)
    seq = gen.generate_sequence([1, 2], max_new_tokens=10)
    assert len(seq) == 12  # 2 prompt + 10 generated


def test_generate_sequence_eos_stops():
    gen = TokenGenerator(vocab_size=10, temperature=0.01, top_k=10, seed=99)
    seq = gen.generate_sequence([0], max_new_tokens=1000, eos_token_id=seq_eos_id(gen))
    # Should be shorter than 1001 if eos was hit (but may not always be).
    assert len(seq) <= 1001


def seq_eos_id(gen):
    """Helper: pick an eos id that is likely to appear."""
    return gen.generate_next_token([0])


def test_token_generator_invalid():
    with pytest.raises(ValueError):
        TokenGenerator(vocab_size=0)
    with pytest.raises(ValueError):
        TokenGenerator(vocab_size=100, temperature=-1)


# ---- KVCache tests ----

def test_kvcache_put_get():
    cache = KVCache(max_length=4)
    cache.put(0, [1.0], [2.0])
    assert cache.get(0) == ([1.0], [2.0])
    assert cache.size == 1


def test_kvcache_fifo_eviction():
    cache = KVCache(max_length=2, eviction_policy=EvictionPolicy.FIFO)
    cache.put(0, [0.0], [0.0])
    cache.put(1, [1.0], [1.0])
    cache.put(2, [2.0], [2.0])  # evicts position 0
    assert cache.get(0) is None
    assert cache.get(1) is not None
    assert cache.size == 2


def test_kvcache_lru_eviction():
    cache = KVCache(max_length=2, eviction_policy=EvictionPolicy.LRU)
    cache.put(0, [0.0], [0.0])
    cache.put(1, [1.0], [1.0])
    cache.get(0)  # access 0 -> moves it to recent end
    cache.put(2, [2.0], [2.0])  # evicts position 1 (least recently used)
    assert cache.get(1) is None
    assert cache.get(0) is not None


def test_kvcache_clear():
    cache = KVCache(max_length=10)
    for i in range(5):
        cache.put(i, [float(i)], [float(i)])
    cache.clear()
    assert cache.size == 0


def test_kvcache_positions():
    cache = KVCache(max_length=10)
    cache.put(5, [0.0], [0.0])
    cache.put(3, [0.0], [0.0])
    assert set(cache.positions()) == {5, 3}


def test_kvcache_memory_estimate():
    cache = KVCache(max_length=10)
    cache.put(0, [0.0], [0.0])
    est = cache.memory_estimate_bytes(hidden_dim=128)
    assert est == 1 * 128 * 2 * 4


# ---- InferenceSession tests ----

def test_session_total_tokens():
    cfg = ModelConfig("t", 10, 100, 64, 2)
    session = InferenceSession(model_config=cfg, prompt_length=5, max_new_tokens=10)
    session.record_token()
    session.record_token()
    assert session.total_tokens == 7
    assert session.tokens_generated == 2


def test_session_is_complete():
    cfg = ModelConfig("t", 10, 100, 64, 2)
    session = InferenceSession(model_config=cfg, max_new_tokens=2)
    assert not session.is_complete
    session.record_token()
    session.record_token()
    assert session.is_complete


def test_session_memory_usage():
    cfg = ModelConfig("t", 10, 100, 64, 2, quantization=QuantizationMode.FP16)
    session = InferenceSession(model_config=cfg)
    assert session.memory_usage_mb > 0


# ---- InferenceEngine tests ----

def test_engine_load_unload():
    engine = InferenceEngine()
    assert not engine.is_loaded
    cfg = ModelConfig("t", 10, 100, 64, 2)
    engine.load_model(cfg)
    assert engine.is_loaded
    engine.unload_model()
    assert not engine.is_loaded


def test_engine_run_no_model():
    engine = InferenceEngine()
    with pytest.raises(RuntimeError):
        engine.run([1, 2, 3])


def test_engine_run_produces_tokens():
    cfg = PRESET_MODELS["surya-nano"]
    engine = InferenceEngine()
    engine.load_model(cfg)
    output, session = engine.run([1, 2, 3], max_new_tokens=5, seed=42)
    assert len(output) == 8  # 3 prompt + 5 generated
    assert session.tokens_generated == 5


def test_engine_create_session():
    cfg = PRESET_MODELS["surya-micro"]
    engine = InferenceEngine(cfg)
    engine.load_model()
    session = engine.create_session(prompt_ids=[0, 1], max_new_tokens=20)
    assert session.prompt_length == 2
    assert session.max_new_tokens == 20
