"""
Dynamic tokenization via Fast Vocabulary Transfer (FVT) from:
  "Retrofitting Large Language Models with Dynamic Tokenization"
  (Feher, Vulic & Minixhofer, 2025) — arXiv:2411.18553, §3.1 & Appendix A

Given a token sequence and a binary boundary mask produced by
boundary_detector.py, this module:
  1. Splits the token sequence into variable-length segments.
  2. For each segment, averages the original subword embeddings (FVT).
  3. Returns a shorter `inputs_embeds` tensor for the frozen LLM.

FVT reference (§2, Gee et al. 2022):
    E_new(t) = (1/|seg|) * sum_{i in seg} E_orig(token_i)

Algorithm 1 (Paper 1 Appendix A) is replicated in `apply_dynamic_tokenization`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def split_by_boundaries(
    input_ids: torch.Tensor,
    boundaries: torch.Tensor,
) -> List[List[int]]:
    """
    Split a 1-D token sequence into segments according to a boundary mask.

    boundaries[t] = True  →  position t is the LAST token of its segment.

    Returns a list of lists; each inner list contains the token-id integers
    for one segment.
    """
    segments: List[List[int]] = []
    current: List[int] = []
    for t, tid in enumerate(input_ids.tolist()):
        current.append(tid)
        if boundaries[t]:
            segments.append(current)
            current = []
    if current:  # flush any trailing tokens (shouldn't happen if boundaries[-1]=True)
        segments.append(current)
    return segments


def fvt_merge(
    segments: List[List[int]],
    embed_table: torch.Tensor,
) -> torch.Tensor:
    """
    Fast Vocabulary Transfer (FVT): for each segment, average the
    embedding-table rows of its constituent token ids.

    Args:
        segments:    list of segments, each a list of token-id ints.
        embed_table: FloatTensor [V, D] — e.g. model.model.embed_tokens.weight

    Returns:
        inputs_embeds: FloatTensor [S, D] where S = len(segments).
    """
    merged = []
    for seg in segments:
        ids = torch.tensor(seg, dtype=torch.long, device=embed_table.device)
        seg_emb = embed_table[ids].mean(dim=0)   # [D]
        merged.append(seg_emb)
    return torch.stack(merged, dim=0)             # [S, D]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_dynamic_tokenization(
    input_ids: torch.Tensor,
    boundaries: torch.Tensor,
    embed_table: torch.Tensor,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Full pipeline: split → FVT merge → return embeddings + merge map.

    This is Algorithm 1 from Paper 1 (Appendix A), adapted to use
    boundary-mask input instead of BPE frequency-based merging.

    Args:
        input_ids:   LongTensor [L]  — original subword token ids.
        boundaries:  BoolTensor [L]  — True = boundary after that position.
        embed_table: FloatTensor [V, D] — model embedding weights.

    Returns:
        inputs_embeds: FloatTensor [S, D]  — merged embeddings (S ≤ L).
        segments:      list[list[int]]     — which original token ids were
                                             merged into each new token.
    """
    segments = split_by_boundaries(input_ids, boundaries)
    inputs_embeds = fvt_merge(segments, embed_table)
    return inputs_embeds, segments


# ---------------------------------------------------------------------------
# Embedding cache (Paper 1 Appendix D — LRU cache)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    LRU cache for FVT embeddings.

    Frequent tokens (e.g. "the", " the") repeat across batches; caching
    their embeddings avoids redundant computation.  The cache key is the
    tuple of token ids in the segment.
    """

    def __init__(self, maxsize: int = 4096):
        self._store: dict = {}
        self._maxsize = maxsize
        self._access: list = []   # LRU tracking

    def get(self, key: Tuple[int, ...]) -> torch.Tensor | None:
        return self._store.get(key, None)

    def set(self, key: Tuple[int, ...], value: torch.Tensor) -> None:
        if key in self._store:
            return
        if len(self._store) >= self._maxsize:
            oldest = self._access.pop(0)
            self._store.pop(oldest, None)
        self._store[key] = value
        self._access.append(key)

    def fvt_with_cache(
        self,
        segments: List[List[int]],
        embed_table: torch.Tensor,
    ) -> torch.Tensor:
        merged = []
        for seg in segments:
            key = tuple(seg)
            cached = self.get(key)
            if cached is None:
                ids = torch.tensor(seg, dtype=torch.long, device=embed_table.device)
                cached = embed_table[ids].mean(dim=0)
                self.set(key, cached)
            merged.append(cached)
        return torch.stack(merged, dim=0)


# ---------------------------------------------------------------------------
# Utility: shortening statistics
# ---------------------------------------------------------------------------

def shortening_factor(original_len: int, merged_len: int) -> float:
    """Return the sequence length reduction ratio (higher = more efficient)."""
    return original_len / merged_len if merged_len > 0 else float("inf")
