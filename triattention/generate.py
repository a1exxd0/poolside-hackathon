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
    peak_kv_len: int                 # max keys stored in a compressed (full-attention) layer
    final_kv_len: int
    num_generated: int
    num_compressions: int
    # Per-decode-step KV memory high-water (bytes, post-append/pre-compression), split
    # by layer kind. Populated only when ``record_kv=True``.
    kv_bytes_full: list[int] = dataclasses.field(default_factory=list)
    kv_bytes_sliding: list[int] = dataclasses.field(default_factory=list)
    kv_bytes_total: list[int] = dataclasses.field(default_factory=list)


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
    record_kv: bool = False,
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
    # Config-aware cache: with mixed attention this gives sliding layers their own
    # window-bounded layer type, so only the full-attention layers grow unbounded.
    cache = DynamicCache(config=model.config)
    L = input_ids.shape[1]

    # Map model-layer-index -> stats for the layers we compress (the full-attention ones).
    stat_for = {}
    if compress:
        stat_for = {li: st for li, st in zip(stats.layer_indices, stats.layers)}
    track = sorted(stat_for) or list(range(len(cache.layers)))   # layers to measure peak on

    def _pos(start, n):
        p = torch.arange(start, start + n, device=device)
        return p, p.unsqueeze(0)

    cache_pos, pos_ids = _pos(0, L)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=cache_pos, position_ids=pos_ids)
    abs_pos = L
    next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
    generated = [next_tok]

    peak_kv = L
    n_compress = 0
    eos = eos_token_id if eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())

    # KV-memory bookkeeping: split layers by kind and size one key/value entry.
    series_full, series_slid, series_tot = [], [], []
    if record_kv:
        lt = getattr(model.config, "layer_types", None) or ["full_attention"] * len(cache.layers)
        full_idx = [i for i, t in enumerate(lt) if t == "full_attention"]
        slid_idx = [i for i, t in enumerate(lt) if t != "full_attention"]
        n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        hd = getattr(model.config, "head_dim", None) or \
            model.config.hidden_size // model.config.num_attention_heads
        itemsize = next(model.parameters()).element_size()
        bytes_per_key = 2 * n_kv * hd * itemsize        # keys + values

        def _record():
            f = sum(cache.layers[i].keys.shape[-2] for i in full_idx)
            s = sum(cache.layers[i].keys.shape[-2] for i in slid_idx)
            series_full.append(f * bytes_per_key)
            series_slid.append(s * bytes_per_key)
            series_tot.append((f + s) * bytes_per_key)

    for step in range(1, max_new_tokens):
        cache_pos, pos_ids = _pos(abs_pos, 1)
        out = model(input_ids=next_tok, past_key_values=cache, use_cache=True,
                    cache_position=cache_pos, position_ids=pos_ids)
        abs_pos += 1
        next_tok = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_tok)

        cur_len = max(cache.layers[li].keys.shape[-2] for li in track)
        peak_kv = max(peak_kv, cur_len)
        if record_kv:
            _record()                                   # post-append high-water for this step
        if next_tok.item() in eos_set:
            break

        if compress and step % beta == 0:
            for li, st in stat_for.items():
                layer = cache.layers[li]
                keys = layer.keys[0]                    # [n_kv, S, d] post-RoPE
                if keys.shape[1] <= budget:
                    continue
                scores = score_keys(keys, abs_pos + 1, st,
                                    group_size=stats.group_size, offsets=offsets)
                _compress_layer(layer, scores, budget, sink)
            n_compress += 1

    seq = torch.cat([input_ids] + generated, dim=1)
    return GenerationResult(
        sequences=seq,
        peak_kv_len=peak_kv,
        final_kv_len=max(cache.layers[li].keys.shape[-2] for li in track),
        num_generated=len(generated),
        num_compressions=n_compress,
        kv_bytes_full=series_full,
        kv_bytes_sliding=series_slid,
        kv_bytes_total=series_tot,
    )
