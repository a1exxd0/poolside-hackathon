"""KV-cache eviction and a compression-aware greedy decode loop.

We drive generation with an explicit ``cache_position`` for every step. This is
what makes importance-based eviction safe: retained keys keep their original
post-RoPE phases (i.e. their absolute positions), and the new query is rotated
with its *true* absolute position, so relative-position geometry is preserved
even though the cache no longer stores a contiguous range of tokens.

Eviction is per (layer, kv-head). Every kv-head keeps exactly ``budget`` keys
(``sink`` leading "attention sink" tokens plus the top ``budget - sink`` by
score), so the cached tensors stay rectangular and ``repeat_kv`` / attention
work unchanged.
"""

from __future__ import annotations

import dataclasses

import torch
from transformers import DynamicCache

from .calibration import CalibrationStats
from .scoring import default_offsets, score_keys


@dataclasses.dataclass
class GenerationResult:
    sequences: torch.Tensor          # [1, prompt + generated]
    peak_kv_len: int                 # max keys stored in any layer during decoding
    final_kv_len: int
    num_generated: int
    num_compressions: int


def _compress_layer(layer, scores: torch.Tensor, budget: int, sink: int) -> None:
    """In-place prune one cache layer to ``budget`` keys per kv-head.

    Args:
        layer: a ``DynamicLayer`` holding ``keys``/``values`` of ``[B, n_kv, S, d]``.
        scores: importance ``[n_kv, S]`` (higher = keep).
    """
    keys, values = layer.keys, layer.values
    b, n_kv, S, d = keys.shape
    if S <= budget:
        return
    scores = scores.clone()
    scores[:, :sink] = float("inf")                     # always retain sink tokens
    keep = scores.topk(budget, dim=-1).indices          # [n_kv, budget]
    keep, _ = keep.sort(dim=-1)                          # keep chronological order
    idx = keep.view(1, n_kv, budget, 1).expand(b, n_kv, budget, d)
    layer.keys = torch.gather(keys, 2, idx)
    layer.values = torch.gather(values, 2, idx)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    stats: CalibrationStats | None = None,
    *,
    max_new_tokens: int = 256,
    budget: int = 2048,
    beta: int = 128,
    sink: int = 4,
    compress: bool = True,
    eos_token_id: int | None = None,
) -> GenerationResult:
    """Greedy-decode with optional TriAttention KV compression.

    Set ``compress=False`` for a full-cache baseline. When ``compress=True``,
    ``stats`` (from :func:`triattention.calibration.collect_calibration`) is
    required.
    """
    if compress and stats is None:
        raise ValueError("compress=True requires calibration stats")

    device = input_ids.device
    offsets = default_offsets(device) if compress else None
    cache = DynamicCache()
    L = input_ids.shape[1]

    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(L, device=device))
    abs_pos = L
    next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
    generated = [next_tok]

    peak_kv = L
    n_compress = 0
    eos = eos_token_id if eos_token_id is not None else getattr(model.config, "eos_token_id", None)

    for step in range(1, max_new_tokens):
        out = model(input_ids=next_tok, past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([abs_pos], device=device))
        abs_pos += 1
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_tok)

        cur_len = cache.layers[0].keys.shape[-2]
        peak_kv = max(peak_kv, cur_len)
        if eos is not None and next_tok.item() == eos:
            break

        if compress and step % beta == 0:
            for li, layer in enumerate(cache.layers):
                keys = layer.keys[0]                    # [n_kv, S, d] post-RoPE
                if keys.shape[1] <= budget:
                    continue
                scores = score_keys(keys, abs_pos + 1, stats.layers[li],
                                    group_size=stats.group_size, offsets=offsets)
                _compress_layer(layer, scores, budget, sink)
            n_compress += 1

    seq = torch.cat([input_ids] + generated, dim=1)
    return GenerationResult(
        sequences=seq,
        peak_kv_len=peak_kv,
        final_kv_len=cache.layers[0].keys.shape[-2],
        num_generated=len(generated),
        num_compressions=n_compress,
    )
