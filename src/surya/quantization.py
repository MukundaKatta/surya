"""
Quantization utilities for Surya inference engine.

Provides memory estimation and speed profiling for different
quantization modes and hardware targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class QuantMode(Enum):
    """Quantization bit-width modes."""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


# Bytes consumed per parameter for each mode.
BYTES_PER_PARAM: Dict[QuantMode, float] = {
    QuantMode.INT4: 0.5,
    QuantMode.INT8: 1.0,
    QuantMode.FP16: 2.0,
    QuantMode.FP32: 4.0,
}


@dataclass
class QuantizationConfig:
    """Describes how a model's weights are quantized.

    Attributes:
        mode: The quantization bit-width.
        group_size: Number of weights sharing a scale factor (for INT4/INT8).
        symmetric: Whether the quantization is symmetric around zero.
    """
    mode: QuantMode
    group_size: int = 128
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

    @property
    def bits(self) -> int:
        """Number of bits per weight element."""
        mapping = {
            QuantMode.INT4: 4,
            QuantMode.INT8: 8,
            QuantMode.FP16: 16,
            QuantMode.FP32: 32,
        }
        return mapping[self.mode]

    @property
    def bytes_per_param(self) -> float:
        """Bytes consumed per model parameter."""
        return BYTES_PER_PARAM[self.mode]

    @property
    def has_scale_overhead(self) -> bool:
        """Whether the mode stores per-group scale factors."""
        return self.mode in (QuantMode.INT4, QuantMode.INT8)

    def overhead_ratio(self) -> float:
        """Ratio of scale-factor overhead relative to raw weight bytes.

        For INT4/INT8 each group stores one FP16 scale (2 bytes).
        Returns 0.0 for floating-point modes.
        """
        if not self.has_scale_overhead:
            return 0.0
        scale_bytes = 2.0  # FP16 scale per group
        group_bytes = self.group_size * self.bytes_per_param
        return scale_bytes / group_bytes

    def effective_bytes_per_param(self) -> float:
        """Bytes per param including quantization overhead."""
        return self.bytes_per_param * (1.0 + self.overhead_ratio())

    def summary(self) -> str:
        """Human-readable description."""
        return (
            f"Quant[{self.mode.value}] "
            f"bits={self.bits} "
            f"group={self.group_size} "
            f"sym={self.symmetric} "
            f"eff_bytes={self.effective_bytes_per_param():.3f}"
        )


class MemoryCalculator:
    """Estimates model memory footprint under various quantization schemes."""

    @staticmethod
    def model_memory_mb(
        params_millions: float,
        quant_config: QuantizationConfig,
    ) -> float:
        """Estimate model weight memory in megabytes.

        Args:
            params_millions: Model size in millions of parameters.
            quant_config: Quantization configuration.

        Returns:
            Estimated memory in MB.
        """
        if params_millions <= 0:
            raise ValueError("params_millions must be positive")
        total_bytes = params_millions * 1e6 * quant_config.effective_bytes_per_param()
        return total_bytes / (1024 * 1024)

    @staticmethod
    def kv_cache_memory_mb(
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
        dtype_bytes: float = 2.0,
    ) -> float:
        """Estimate KV-cache memory for all layers.

        Each layer stores keys and values (2 tensors), each of shape
        (max_seq_len, hidden_dim).

        Args:
            num_layers: Number of transformer layers.
            hidden_dim: Hidden dimension size.
            max_seq_len: Maximum sequence length.
            dtype_bytes: Bytes per element (default FP16 = 2).

        Returns:
            Estimated KV-cache memory in MB.
        """
        # 2 tensors (K, V) per layer.
        total_bytes = num_layers * 2 * max_seq_len * hidden_dim * dtype_bytes
        return total_bytes / (1024 * 1024)

    @staticmethod
    def total_memory_mb(
        params_millions: float,
        quant_config: QuantizationConfig,
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> float:
        """Total estimated memory (model weights + KV cache)."""
        model = MemoryCalculator.model_memory_mb(params_millions, quant_config)
        cache = MemoryCalculator.kv_cache_memory_mb(
            num_layers, hidden_dim, max_seq_len
        )
        return model + cache


class HardwareProfile:
    """Simple hardware profile for speed estimation.

    Attributes:
        name: Profile name (e.g. "rpi4", "jetson-nano").
        memory_bandwidth_gbps: Memory bandwidth in GB/s.
        compute_tops: Compute throughput in tera-ops/second.
    """

    def __init__(
        self,
        name: str,
        memory_bandwidth_gbps: float,
        compute_tops: float,
    ) -> None:
        if memory_bandwidth_gbps <= 0:
            raise ValueError("memory_bandwidth_gbps must be positive")
        if compute_tops <= 0:
            raise ValueError("compute_tops must be positive")
        self.name = name
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self.compute_tops = compute_tops


# Built-in hardware profiles.
HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "rpi4": HardwareProfile("rpi4", memory_bandwidth_gbps=4.0, compute_tops=0.013),
    "jetson-nano": HardwareProfile("jetson-nano", memory_bandwidth_gbps=25.6, compute_tops=0.472),
    "apple-m1": HardwareProfile("apple-m1", memory_bandwidth_gbps=68.25, compute_tops=11.0),
}


class SpeedEstimator:
    """Estimates tokens-per-second for a model on a given hardware profile.

    Uses a simplified roofline model: generation is bottlenecked by
    whichever is slower between memory bandwidth and compute throughput.
    """

    @staticmethod
    def estimate_tokens_per_second(
        params_millions: float,
        quant_config: QuantizationConfig,
        hardware: HardwareProfile,
    ) -> float:
        """Estimate peak tokens/second.

        For each generated token the model weights are read once
        (memory-bound) and ~2 * params FLOPs are executed (compute-bound).

        Args:
            params_millions: Model size in millions of parameters.
            quant_config: Quantization config.
            hardware: Target hardware profile.

        Returns:
            Estimated tokens per second.
        """
        if params_millions <= 0:
            raise ValueError("params_millions must be positive")

        weight_bytes = params_millions * 1e6 * quant_config.effective_bytes_per_param()
        bandwidth_bytes_per_sec = hardware.memory_bandwidth_gbps * 1e9

        # Memory-bound: how fast we can stream the weights.
        mem_tok_per_sec = bandwidth_bytes_per_sec / weight_bytes

        # Compute-bound: 2 * params FLOPs per token.
        flops_per_token = 2.0 * params_millions * 1e6
        compute_flops_per_sec = hardware.compute_tops * 1e12
        compute_tok_per_sec = compute_flops_per_sec / flops_per_token

        # Roofline: bottleneck is the slower path.
        return min(mem_tok_per_sec, compute_tok_per_sec)

    @staticmethod
    def compare_hardware(
        params_millions: float,
        quant_config: QuantizationConfig,
        profiles: Optional[Dict[str, HardwareProfile]] = None,
    ) -> Dict[str, float]:
        """Compare tokens/second across multiple hardware profiles.

        Args:
            params_millions: Model size.
            quant_config: Quantization config.
            profiles: Hardware profiles to compare. Defaults to built-ins.

        Returns:
            Dict mapping profile name to estimated tokens/second.
        """
        if profiles is None:
            profiles = HARDWARE_PROFILES
        return {
            name: SpeedEstimator.estimate_tokens_per_second(
                params_millions, quant_config, hw
            )
            for name, hw in profiles.items()
        }
