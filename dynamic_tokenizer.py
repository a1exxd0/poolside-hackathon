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

import random
from collections import Counter, OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_SIZE = 4096

# Token prefixes that mark the start of a new word / pre-token.
# GPT-2/Llama BPE: Ġ (space), Ċ (newline); SentencePiece: ▁.
_WORD_START_PREFIXES = (" ", "\n", "Ġ", "Ċ", "▁")


# ---------------------------------------------------------------------------
# BPE-style dynamic tokenization (§3.1 & Appendix A, Feher et al. 2025)
# ---------------------------------------------------------------------------

def is_word_start(token_str: str) -> bool:
    """True if token_str begins a new pre-token (word boundary)."""
    return bool(token_str) and token_str.startswith(_WORD_START_PREFIXES)


def get_word_boundary_mask(input_ids: torch.Tensor, tokenizer) -> List[bool]:
    """
    For each position i, True means token i starts a new word and must not be
    merged with token i-1.  Position 0 is always False.

    Special tokens and positions immediately adjacent to them are also marked
    True to prevent merging into or out of special tokens.
    """
    ids_list = input_ids.tolist()
    token_strings = tokenizer.convert_ids_to_tokens(ids_list)
    special_ids = set(tokenizer.all_special_ids)
    mask = [False] * len(ids_list)
    for i in range(1, len(ids_list)):
        s = token_strings[i] or ""
        if (
            is_word_start(s)
            or ids_list[i] in special_ids
            or ids_list[i - 1] in special_ids
        ):
            mask[i] = True
    return mask


def compute_mmax(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Upper-bound on merge steps to reach word-level tokenization.

    Equals the number of within-word adjacent token pairs in the sequence —
    i.e. positions i where mask[i] is False and the pair (i-1, i) can be merged.
    """
    mask = get_word_boundary_mask(input_ids, tokenizer)
    return sum(1 for i in range(1, len(mask)) if not mask[i])


def _apply_merge_to_sequence(
    segs: List[List[int]],
    first_origs: List[int],
    best_pair: Tuple[Tuple[int, ...], Tuple[int, ...]],
    word_boundary_mask: List[bool],
) -> Tuple[List[List[int]], List[int]]:
    """Apply one BPE merge step to a single tokenised sequence."""
    new_segs: List[List[int]] = []
    new_firsts: List[int] = []
    j = 0
    while j < len(segs):
        if (
            j + 1 < len(segs)
            and tuple(segs[j]) == best_pair[0]
            and tuple(segs[j + 1]) == best_pair[1]
            and not word_boundary_mask[first_origs[j + 1]]
        ):
            new_segs.append(segs[j] + segs[j + 1])
            new_firsts.append(first_origs[j])
            j += 2
        else:
            new_segs.append(segs[j])
            new_firsts.append(first_origs[j])
            j += 1
    return new_segs, new_firsts


def bpe_dynamic_tokenize(
    input_ids_batch: List[torch.Tensor],
    tokenizer,
    m: int,
) -> List[List[List[int]]]:
    """
    Batch-level BPE-style dynamic tokenization — Algorithm 1, Feher et al. 2025.

    Each merge step counts adjacent segment-pair frequencies across the full
    batch (never crossing word boundaries), picks the most frequent pair, and
    merges every occurrence of it in all sequences.

    Args:
        input_ids_batch: one 1-D LongTensor per sequence.
        tokenizer:        HuggingFace tokenizer.
        m:               number of BPE merge operations.

    Returns:
        segments_batch[b][j] = list of original token ids merged into segment j
        of sequence b.
    """
    ids_lists = [ids.tolist() for ids in input_ids_batch]
    seg_sequences: List[List[List[int]]] = [[[tok] for tok in ids] for ids in ids_lists]
    seg_first_origs: List[List[int]] = [[i for i in range(len(ids))] for ids in ids_lists]
    word_boundary_masks = [get_word_boundary_mask(ids, tokenizer) for ids in input_ids_batch]

    for _step in range(m):
        pair_counts: Counter = Counter()
        for b in range(len(seg_sequences)):
            segs = seg_sequences[b]
            firsts = seg_first_origs[b]
            wbm = word_boundary_masks[b]
            for j in range(len(segs) - 1):
                if not wbm[firsts[j + 1]]:
                    pair = (tuple(segs[j]), tuple(segs[j + 1]))
                    pair_counts[pair] += 1

        if not pair_counts:
            break  # word-level tokenization reached; no more within-word pairs

        best_pair = pair_counts.most_common(1)[0][0]
        for b in range(len(seg_sequences)):
            seg_sequences[b], seg_first_origs[b] = _apply_merge_to_sequence(
                seg_sequences[b], seg_first_origs[b], best_pair, word_boundary_masks[b]
            )

    return seg_sequences


def apply_dynamic_tokenization_bpe(
    input_ids_batch: List[torch.Tensor],
    tokenizer,
    embed_table: torch.Tensor,
    m: int | None = None,
    sample_merges: bool = False,
    cache: "EmbeddingCache | None" = None,
) -> Tuple[List[torch.Tensor], List[List[List[int]]], int]:
    """
    Full BPE-style dynamic tokenization pipeline (§3.1, Feher et al. 2025).

    1. Compute mmax (within-word adjacent pairs) per sequence; take the min.
    2. Determine m: explicit value, sampled from U(0, mmax), or 50% of mmax.
    3. Run batch-level BPE merging → segments.
    4. Apply FVT embeddings per sequence (via cache if supplied).

    Args:
        input_ids_batch: list of 1-D LongTensors.
        tokenizer:        HuggingFace tokenizer.
        embed_table:      FloatTensor [V, D].
        m:               explicit merge count; None → 50 % of mmax.
        sample_merges:   sample m ~ U(0, mmax) (paper eq. 4).
        cache:           optional EmbeddingCache.

    Returns:
        (inputs_embeds_batch, segments_batch, m_used)
    """
    mmax_vals = [compute_mmax(ids, tokenizer) for ids in input_ids_batch]
    mmax = min(mmax_vals) if mmax_vals else 0

    if sample_merges:
        m_used = random.randint(0, mmax)
    elif m is None:
        m_used = max(1, mmax // 2)
    else:
        m_used = min(m, mmax)

    segments_batch = bpe_dynamic_tokenize(input_ids_batch, tokenizer, m_used)

    inputs_embeds_batch: List[torch.Tensor] = []
    for segs in segments_batch:
        if cache is not None:
            embeds = cache.fvt_with_cache(segs, embed_table)
        else:
            embeds = fvt_merge(segs, embed_table)
        inputs_embeds_batch.append(embeds)

    return inputs_embeds_batch, segments_batch, m_used


# ---------------------------------------------------------------------------
# Core helpers (legacy boundary-mask path)
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
    D = embed_table.shape[1]
    out = torch.empty(len(segments), D, dtype=embed_table.dtype, device=embed_table.device)
    for i, seg in enumerate(segments):
        ids = torch.tensor(seg, dtype=torch.long, device=embed_table.device)
        out[i] = embed_table[ids].mean(dim=0)
    return out


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

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE):
        self._store: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: Tuple[int, ...]) -> torch.Tensor | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: Tuple[int, ...], value: torch.Tensor) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)  # evict least recently used
        self._store[key] = value

    def fvt_with_cache(
        self,
        segments: List[List[int]],
        embed_table: torch.Tensor,
    ) -> torch.Tensor:
        D = embed_table.shape[1]
        out = torch.empty(len(segments), D, dtype=embed_table.dtype, device=embed_table.device)
        for i, seg in enumerate(segments):
            key = tuple(seg)
            cached = self.get(key)
            if cached is None:
                ids = torch.tensor(seg, dtype=torch.long, device=embed_table.device)
                cached = embed_table[ids].mean(dim=0)
                self.set(key, cached)
            out[i] = cached
        return out


# ---------------------------------------------------------------------------
# Utility: shortening statistics
# ---------------------------------------------------------------------------

def shortening_factor(original_len: int, merged_len: int) -> float:
    """Return the sequence length reduction ratio (higher = more efficient)."""
    return original_len / merged_len if merged_len > 0 else float("inf")
