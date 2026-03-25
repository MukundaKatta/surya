"""
Core inference engine for Surya - Tiny LLM Inference Engine.

Provides simulated LLM inference with model loading, token generation,
KV-cache management, and session tracking for embedded devices.
"""

from __future__ import annotations

import random
import time
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class QuantizationMode(Enum):
    """Supported quantization modes for model weights."""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class EvictionPolicy(Enum):
    """Cache eviction policies for the KV cache."""
    FIFO = "fifo"
    LRU = "lru"


@dataclass
class ModelConfig:
    """Configuration for a simulated LLM model.

    Attributes:
        name: Human-readable model name.
        params_millions: Number of parameters in millions.
        vocab_size: Size of the token vocabulary.
        hidden_dim: Dimension of the hidden layers.
        num_layers: Number of transformer layers.
        quantization: Quantization mode for model weights.
        max_sequence_length: Maximum supported sequence length.
    """
    name: str
    params_millions: float
    vocab_size: int
    hidden_dim: int
    num_layers: int
    quantization: QuantizationMode = QuantizationMode.FP16
    max_sequence_length: int = 2048

    def __post_init__(self) -> None:
        if self.params_millions <= 0:
            raise ValueError("params_millions must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

    @property
    def bytes_per_param(self) -> float:
        """Return bytes per parameter based on quantization mode."""
        mapping = {
            QuantizationMode.INT4: 0.5,
            QuantizationMode.INT8: 1.0,
            QuantizationMode.FP16: 2.0,
            QuantizationMode.FP32: 4.0,
        }
        return mapping[self.quantization]

    @property
    def estimated_memory_mb(self) -> float:
        """Estimate total model memory in megabytes."""
        total_bytes = self.params_millions * 1e6 * self.bytes_per_param
        return total_bytes / (1024 * 1024)

    def summary(self) -> str:
        """Return a human-readable summary of the model config."""
        return (
            f"Model: {self.name} | "
            f"{self.params_millions}M params | "
            f"vocab={self.vocab_size} | "
            f"hidden={self.hidden_dim} | "
            f"layers={self.num_layers} | "
            f"quant={self.quantization.value} | "
            f"~{self.estimated_memory_mb:.1f} MB"
        )


# Predefined model configurations for common tiny models.
PRESET_MODELS: Dict[str, ModelConfig] = {
    "surya-nano": ModelConfig(
        name="surya-nano",
        params_millions=25,
        vocab_size=8000,
        hidden_dim=256,
        num_layers=6,
        quantization=QuantizationMode.INT4,
    ),
    "surya-micro": ModelConfig(
        name="surya-micro",
        params_millions=70,
        vocab_size=16000,
        hidden_dim=512,
        num_layers=12,
        quantization=QuantizationMode.INT8,
    ),
    "surya-mini": ModelConfig(
        name="surya-mini",
        params_millions=150,
        vocab_size=32000,
        hidden_dim=768,
        num_layers=12,
        quantization=QuantizationMode.FP16,
    ),
}


class TokenGenerator:
    """Simulated token generator with temperature and top-k sampling.

    This does NOT run a real model; it draws from a probability distribution
    over the vocabulary to simulate autoregressive generation.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        temperature: Sampling temperature (higher = more random).
        top_k: Number of top candidates to consider during sampling.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        vocab_size: int,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k <= 0 or top_k > vocab_size:
            raise ValueError("top_k must be between 1 and vocab_size")

        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_k = top_k
        self._rng = random.Random(seed)

    def _softmax(self, logits: List[float]) -> List[float]:
        """Apply softmax to a list of logits."""
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _apply_temperature(self, logits: List[float]) -> List[float]:
        """Scale logits by the temperature."""
        return [l / self.temperature for l in logits]

    def _top_k_filter(
        self, logits: List[float]
    ) -> List[Tuple[int, float]]:
        """Return the top-k (token_id, logit) pairs."""
        indexed = list(enumerate(logits))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[: self.top_k]

    def generate_next_token(self, context_ids: Optional[List[int]] = None) -> int:
        """Generate the next token id given optional context.

        The simulation creates random logits, applies temperature scaling
        and top-k filtering, then samples from the resulting distribution.

        Args:
            context_ids: Previously generated token ids (used to seed
                         logit generation for reproducibility).

        Returns:
            A sampled token id in [0, vocab_size).
        """
        # Simulate logits as random values influenced by context length.
        logits = [self._rng.gauss(0, 1) for _ in range(self.vocab_size)]

        # Bias toward lower-id tokens slightly (simulates frequency bias).
        if context_ids:
            for tid in context_ids[-5:]:
                if 0 <= tid < self.vocab_size:
                    logits[tid] += 0.3

        scaled = self._apply_temperature(logits)
        top_k_pairs = self._top_k_filter(scaled)

        top_k_ids = [tid for tid, _ in top_k_pairs]
        top_k_logits = [l for _, l in top_k_pairs]
        probs = self._softmax(top_k_logits)

        r = self._rng.random()
        cumulative = 0.0
        for tid, p in zip(top_k_ids, probs):
            cumulative += p
            if r <= cumulative:
                return tid
        return top_k_ids[-1]

    def generate_sequence(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Generate a sequence of tokens autoregressively.

        Args:
            prompt_ids: Input token ids to condition on.
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: If generated, stops early.

        Returns:
            The full sequence (prompt + generated tokens).
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        result = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = self.generate_next_token(result)
            result.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
        return result


class KVCache:
    """Key-value cache for transformer attention layers.

    Simulates a bounded cache with configurable eviction policy.

    Attributes:
        max_length: Maximum number of entries the cache can hold.
        eviction_policy: Strategy for evicting entries when full.
    """

    def __init__(
        self,
        max_length: int = 512,
        eviction_policy: EvictionPolicy = EvictionPolicy.FIFO,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.max_length = max_length
        self.eviction_policy = eviction_policy
        self._cache: OrderedDict[int, Tuple[List[float], List[float]]] = (
            OrderedDict()
        )

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    @property
    def is_full(self) -> bool:
        """Whether the cache has reached its maximum capacity."""
        return self.size >= self.max_length

    def put(self, position: int, key: List[float], value: List[float]) -> None:
        """Insert or update a cache entry at the given position.

        If the cache is full, the eviction policy determines which
        entry is removed first.

        Args:
            position: Sequence position for this KV pair.
            key: The key vector (simulated).
            value: The value vector (simulated).
        """
        if position in self._cache:
            # Move to end on update (relevant for LRU).
            self._cache.move_to_end(position)
            self._cache[position] = (key, value)
            return

        while self.is_full:
            self._evict()

        self._cache[position] = (key, value)

    def get(self, position: int) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieve the KV pair for a given position.

        For LRU policy, accessing an entry moves it to the most-recent end.

        Args:
            position: Sequence position to look up.

        Returns:
            (key, value) tuple if found, else None.
        """
        if position not in self._cache:
            return None

        if self.eviction_policy == EvictionPolicy.LRU:
            self._cache.move_to_end(position)

        return self._cache[position]

    def _evict(self) -> None:
        """Evict one entry according to the eviction policy."""
        if not self._cache:
            return
        # Both FIFO and LRU evict from the front (oldest / least-recently-used).
        self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    def positions(self) -> List[int]:
        """Return all cached positions in insertion order."""
        return list(self._cache.keys())

    def memory_estimate_bytes(self, hidden_dim: int = 256) -> int:
        """Estimate the memory footprint of the cache in bytes.

        Each entry stores two vectors (key, value) of floats.

        Args:
            hidden_dim: Dimension of key/value vectors.

        Returns:
            Estimated bytes consumed.
        """
        # 4 bytes per float, 2 vectors per entry.
        return self.size * hidden_dim * 2 * 4


@dataclass
class InferenceSession:
    """Tracks the state of a single inference run.

    Attributes:
        model_config: The model being used.
        tokens_generated: Running count of tokens produced so far.
        start_time: Epoch time when the session started.
        kv_cache: The cache instance bound to this session.
        prompt_length: Number of tokens in the initial prompt.
        max_new_tokens: Cap on new tokens to generate.
    """
    model_config: ModelConfig
    tokens_generated: int = 0
    start_time: float = field(default_factory=time.time)
    kv_cache: Optional[KVCache] = None
    prompt_length: int = 0
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.kv_cache is None:
            self.kv_cache = KVCache(
                max_length=self.model_config.max_sequence_length
            )

    @property
    def total_tokens(self) -> int:
        """Total tokens including prompt and generated."""
        return self.prompt_length + self.tokens_generated

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the session started."""
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Measured generation throughput."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.tokens_generated / elapsed

    @property
    def memory_usage_mb(self) -> float:
        """Estimated total memory usage in megabytes.

        Includes model weights plus the KV cache footprint.
        """
        model_mb = self.model_config.estimated_memory_mb
        cache_bytes = 0
        if self.kv_cache is not None:
            cache_bytes = self.kv_cache.memory_estimate_bytes(
                self.model_config.hidden_dim
            )
        return model_mb + cache_bytes / (1024 * 1024)

    @property
    def is_complete(self) -> bool:
        """Whether the session has reached its generation limit."""
        return self.tokens_generated >= self.max_new_tokens

    def record_token(self) -> None:
        """Increment the generated-token counter by one."""
        self.tokens_generated += 1

    def summary(self) -> str:
        """Return a human-readable session summary."""
        return (
            f"Session[{self.model_config.name}]: "
            f"prompt={self.prompt_length}, "
            f"generated={self.tokens_generated}/{self.max_new_tokens}, "
            f"mem~{self.memory_usage_mb:.1f}MB, "
            f"{self.tokens_per_second:.1f} tok/s"
        )


class InferenceEngine:
    """High-level inference engine that ties everything together.

    Provides a simple API for loading a model config, starting an
    inference session, and running simulated token generation.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self._model_config = model_config
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether a model has been loaded."""
        return self._loaded

    @property
    def model_config(self) -> Optional[ModelConfig]:
        """The currently loaded model config, or None."""
        return self._model_config

    def load_model(self, config: Optional[ModelConfig] = None) -> ModelConfig:
        """Simulate loading model weights into memory.

        Args:
            config: Model configuration. If None, uses the one passed
                    at construction time.

        Returns:
            The active ModelConfig.

        Raises:
            ValueError: If no config is available.
        """
        if config is not None:
            self._model_config = config
        if self._model_config is None:
            raise ValueError("No model config provided")
        self._loaded = True
        return self._model_config

    def unload_model(self) -> None:
        """Simulate unloading the model from memory."""
        self._loaded = False

    def create_session(
        self,
        prompt_ids: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        cache_eviction: EvictionPolicy = EvictionPolicy.FIFO,
    ) -> InferenceSession:
        """Create a new inference session.

        Args:
            prompt_ids: Token ids for the prompt.
            max_new_tokens: Maximum tokens to generate.
            cache_eviction: KV-cache eviction policy.

        Returns:
            A new InferenceSession.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self._loaded or self._model_config is None:
            raise RuntimeError("Model must be loaded before creating a session")

        prompt_len = len(prompt_ids) if prompt_ids else 0
        kv_cache = KVCache(
            max_length=self._model_config.max_sequence_length,
            eviction_policy=cache_eviction,
        )
        return InferenceSession(
            model_config=self._model_config,
            prompt_length=prompt_len,
            max_new_tokens=max_new_tokens,
            kv_cache=kv_cache,
        )

    def run(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], InferenceSession]:
        """Run end-to-end simulated inference.

        Args:
            prompt_ids: Input token ids.
            max_new_tokens: Tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k candidates.
            eos_token_id: Optional early-stop token.
            seed: RNG seed for reproducibility.

        Returns:
            (generated_ids, session) tuple.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self._loaded or self._model_config is None:
            raise RuntimeError("Model must be loaded before running inference")

        session = self.create_session(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )

        generator = TokenGenerator(
            vocab_size=self._model_config.vocab_size,
            temperature=temperature,
            top_k=min(top_k, self._model_config.vocab_size),
            seed=seed,
        )

        context = list(prompt_ids)
        for _ in range(max_new_tokens):
            token_id = generator.generate_next_token(context)
            context.append(token_id)
            session.record_token()

            # Simulate populating the KV cache.
            pos = len(context) - 1
            dummy_vec = [0.0] * self._model_config.hidden_dim
            if session.kv_cache is not None:
                session.kv_cache.put(pos, dummy_vec, dummy_vec)

            if eos_token_id is not None and token_id == eos_token_id:
                break

        return context, session
