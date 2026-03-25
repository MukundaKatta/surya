# 🔱 Surya — Tiny LLM Inference Engine

> **Hindu Mythology**: The Sun God | Lightweight inference engine for edge devices

[![GitHub Pages](https://img.shields.io/badge/🌐_Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/surya/)
[![GitHub](https://img.shields.io/github/license/MukundaKatta/surya?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/MukundaKatta/surya?style=flat-square)](https://github.com/MukundaKatta/surya/stargazers)

## Overview

Surya is a simulated tiny LLM inference engine designed for embedded and edge devices. It provides model loading simulation, autoregressive token generation, KV-cache management with eviction policies, quantization configuration, and hardware speed estimation — all with zero external dependencies.

**Tech Stack:** Python 3.9+

## Quick Start

```bash
git clone https://github.com/MukundaKatta/surya.git
cd surya
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Usage

```python
from surya.core import InferenceEngine, PRESET_MODELS

engine = InferenceEngine()
engine.load_model(PRESET_MODELS["surya-nano"])

output_ids, session = engine.run(
    prompt_ids=[1, 2, 3],
    max_new_tokens=16,
    temperature=0.8,
    top_k=40,
    seed=42,
)
print(session.summary())
```

## Project Structure

```
surya/
├── src/
│   └── surya/
│       ├── __init__.py
│       ├── core.py            # InferenceEngine, TokenGenerator, KVCache, ModelConfig
│       ├── quantization.py    # QuantizationConfig, MemoryCalculator, SpeedEstimator
│       └── tokenizer.py       # SimpleTokenizer, Vocabulary, SpecialTokens
├── tests/
│   ├── test_core.py
│   ├── test_quantization.py
│   └── test_tokenizer.py
├── README.md
├── LICENSE
└── CLAUDE.md
```

## Features

- **Model Configuration** — Dataclass-based configs with preset tiny models (25M–150M params)
- **Token Generation** — Temperature and top-k sampling over simulated logit distributions
- **KV-Cache** — Bounded cache with FIFO and LRU eviction policies
- **Quantization** — INT4, INT8, FP16, FP32 modes with overhead-aware memory estimation
- **Speed Estimation** — Roofline-model speed estimates across hardware profiles (RPi4, Jetson Nano, Apple M1)
- **Tokenizer** — Word-level tokenizer with BOS/EOS/PAD/UNK special token handling

## Live Demo

Visit the landing page: **https://MukundaKatta.github.io/surya/**

## License

MIT License — © 2026 Officethree Technologies

## 🔱 Part of the Mythological Portfolio

This is project **#surya** in the [100-project Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
