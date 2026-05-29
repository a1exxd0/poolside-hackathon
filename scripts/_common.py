"""Shared helpers for the calibration / validation scripts."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small reasoning model that fits in 18 GB unified memory and exercises the GQA
# + RoPE path (Qwen2: 12 query heads / 2 kv heads, head_dim 128).
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: str | None = None, dtype=torch.float16):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device).eval()
    return model, tok, device


# A few generic passages for offline calibration of pre-RoPE Q/K statistics.
CALIBRATION_TEXTS = [
    "The process of photosynthesis converts light energy into chemical energy stored "
    "in glucose. Chlorophyll in the thylakoid membranes absorbs photons, exciting "
    "electrons that drive the synthesis of ATP and NADPH during the light reactions.",
    "In number theory, a prime number is a natural number greater than one that has no "
    "positive divisors other than one and itself. The fundamental theorem of arithmetic "
    "states that every integer greater than one is either prime or a product of primes.",
    "To solve a quadratic equation of the form a x squared plus b x plus c equals zero, "
    "one may apply the quadratic formula, completing the square, or factoring. The "
    "discriminant b squared minus four a c determines the nature of the roots.",
    "The French Revolution began in 1789 amid widespread social and economic discontent. "
    "It dismantled the absolute monarchy, reshaped political institutions, and influenced "
    "the spread of liberal and nationalist ideas across Europe in the decades that followed.",
    "A binary search algorithm repeatedly divides a sorted array in half to locate a "
    "target value. Each comparison eliminates half of the remaining elements, giving a "
    "worst-case time complexity that is logarithmic in the size of the input array.",
]
