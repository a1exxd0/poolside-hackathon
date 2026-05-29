"""TriAttention: trigonometric KV-cache compression for long reasoning.

Implementation of arXiv:2604.04921v1 — "TriAttention: Efficient Long Reasoning
with Trigonometric KV Compression".

Pipeline:
    1. ``collect_calibration`` — estimate pre-RoPE Q/K band statistics offline.
    2. ``score_keys`` — score cached keys by predicted future importance.
    3. ``generate`` — greedy decode that prunes the KV cache to a fixed budget.
"""

from .calibration import CalibrationStats, LayerStats, collect_calibration
from .generate import GenerationResult, generate
from .rope import rope_frequencies, to_complex_bands
from .scoring import default_offsets, score_keys

__all__ = [
    "CalibrationStats",
    "LayerStats",
    "collect_calibration",
    "GenerationResult",
    "generate",
    "rope_frequencies",
    "to_complex_bands",
    "default_offsets",
    "score_keys",
]
