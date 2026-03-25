"""Tests for surya.quantization — memory and speed estimation."""

import pytest

from surya.quantization import (
    HARDWARE_PROFILES,
    HardwareProfile,
    MemoryCalculator,
    QuantizationConfig,
    QuantMode,
    SpeedEstimator,
)


# ---- QuantizationConfig tests ----

def test_quant_config_bits():
    assert QuantizationConfig(QuantMode.INT4).bits == 4
    assert QuantizationConfig(QuantMode.FP32).bits == 32


def test_quant_config_bytes_per_param():
    assert QuantizationConfig(QuantMode.INT4).bytes_per_param == 0.5
    assert QuantizationConfig(QuantMode.FP16).bytes_per_param == 2.0


def test_quant_config_overhead():
    q4 = QuantizationConfig(QuantMode.INT4, group_size=128)
    assert q4.overhead_ratio() > 0
    fp32 = QuantizationConfig(QuantMode.FP32)
    assert fp32.overhead_ratio() == 0.0


def test_quant_config_invalid_group():
    with pytest.raises(ValueError):
        QuantizationConfig(QuantMode.INT8, group_size=0)


def test_quant_config_effective_bytes():
    fp16 = QuantizationConfig(QuantMode.FP16)
    assert fp16.effective_bytes_per_param() == 2.0  # no overhead
    int4 = QuantizationConfig(QuantMode.INT4, group_size=64)
    assert int4.effective_bytes_per_param() > int4.bytes_per_param


def test_quant_config_summary():
    q = QuantizationConfig(QuantMode.INT8)
    s = q.summary()
    assert "int8" in s


# ---- MemoryCalculator tests ----

def test_model_memory_fp32():
    q = QuantizationConfig(QuantMode.FP32)
    mb = MemoryCalculator.model_memory_mb(100, q)
    assert mb == pytest.approx(381.47, rel=0.01)


def test_model_memory_int4_less_than_fp32():
    q4 = QuantizationConfig(QuantMode.INT4)
    q32 = QuantizationConfig(QuantMode.FP32)
    assert MemoryCalculator.model_memory_mb(50, q4) < MemoryCalculator.model_memory_mb(50, q32)


def test_kv_cache_memory():
    mb = MemoryCalculator.kv_cache_memory_mb(12, 256, 1024, 2.0)
    assert mb > 0


def test_total_memory():
    q = QuantizationConfig(QuantMode.FP16)
    total = MemoryCalculator.total_memory_mb(50, q, 12, 256, 1024)
    model_only = MemoryCalculator.model_memory_mb(50, q)
    assert total > model_only


def test_model_memory_invalid():
    q = QuantizationConfig(QuantMode.FP16)
    with pytest.raises(ValueError):
        MemoryCalculator.model_memory_mb(-10, q)


# ---- SpeedEstimator tests ----

def test_speed_estimator_positive():
    q = QuantizationConfig(QuantMode.FP16)
    hw = HARDWARE_PROFILES["apple-m1"]
    tps = SpeedEstimator.estimate_tokens_per_second(70, q, hw)
    assert tps > 0


def test_speed_int4_faster_than_fp32():
    q4 = QuantizationConfig(QuantMode.INT4)
    q32 = QuantizationConfig(QuantMode.FP32)
    hw = HARDWARE_PROFILES["rpi4"]
    tps4 = SpeedEstimator.estimate_tokens_per_second(50, q4, hw)
    tps32 = SpeedEstimator.estimate_tokens_per_second(50, q32, hw)
    assert tps4 > tps32


def test_compare_hardware():
    q = QuantizationConfig(QuantMode.INT8)
    results = SpeedEstimator.compare_hardware(100, q)
    assert len(results) == len(HARDWARE_PROFILES)
    for name, tps in results.items():
        assert tps > 0
