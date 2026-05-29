"""TriAttention importance scoring.

Given the post-RoPE keys currently in the cache and the calibrated query
statistics, estimate how important each key will be to *future* queries, then
return one score per (kv-head, key).

Key identity used here (see :mod:`triattention.rope`):

    logit(q, k) = sum_f |z^q_f| |z^k_f| cos(arg z^q_f - arg z^k_f + omega_f * (p_q - p_k))

Replacing the unknown future query band ``z^q_f`` with its calibrated centre
``E[q_f]`` and folding the key's absolute position into its *post-RoPE* angle
(``arg z^k_post_f = arg z^k_pre_f + p_k * omega_f``) gives the trigonometric
score, evaluated directly from cached keys:

    S_trig(k) = mean_{delta in D} sum_f |E[q_f]| |k_f| cos(omega_f (p_q + delta) + arg E[q_f] - arg k_post_f)

The norm score captures the importance contributed by query *variation* around
that centre (large when the band is poorly concentrated, R_f small):

    S_norm(k) = sum_f (1 - R_f) E[|q_f|] |k_f|
"""

from __future__ import annotations

import torch

from .calibration import LayerStats
from .rope import to_complex_bands, pass_through_dims


def default_offsets(device=None) -> torch.Tensor:
    """``D = {2^0, 2^1, ..., 2^16}`` — log-spaced future query distances."""
    return (2 ** torch.arange(0, 17, device=device)).float()


def per_head_scores(
    keys_post: torch.Tensor,
    query_pos: int,
    stats: LayerStats,
    *,
    group_size: int,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return raw per-query-head scores ``[n_q, S]`` (before GQA aggregation).

    ``S(k) = S_trig(k) + S_norm(k)``, with ``S_trig`` averaged over the future
    offsets ``D``. Exposed separately so the reconstruction property can be
    validated without the z-score/max aggregation.
    """
    n_kv, S, d = keys_post.shape
    device = keys_post.device
    if offsets is None:
        offsets = default_offsets(device)
    offsets = offsets.to(device=device, dtype=torch.float32)

    n_q = stats.Eq.shape[0]
    K = stats.dominant_bands.shape[1]
    dom = stats.dominant_bands                              # [n_q, K]
    rotary_dim = getattr(stats, "rotary_dim", None) or d
    kv_for_q = torch.arange(n_q, device=device) // group_size      # [n_q]

    # --- query-side gathered quantities (per query head, dominant bands) ---
    Eq_mag = stats.Eq.abs().gather(1, dom)                 # [n_q, K]
    Eq_arg = stats.Eq.angle().gather(1, dom)               # [n_q, K]
    Eq_norm = stats.Eq_norm.gather(1, dom)                 # [n_q, K]
    R = stats.R.gather(1, dom)                             # [n_q, K]
    omega = stats.omega.unsqueeze(0).expand(n_q, -1).gather(1, dom)  # [n_q, K]

    # --- key-side gathered quantities (per query head's kv head, dominant bands) ---
    kbands = to_complex_bands(keys_post, rotary_dim)       # [n_kv, S, d2] complex
    kmag = kbands.abs()                                    # [n_kv, S, d2]
    karg = kbands.angle()
    kmag_q = kmag[kv_for_q]                                # [n_q, S, d2]
    karg_q = karg[kv_for_q]
    dom_S = dom.unsqueeze(1).expand(n_q, S, K)
    kmag_sel = torch.gather(kmag_q, 2, dom_S)              # [n_q, S, K]
    karg_sel = torch.gather(karg_q, 2, dom_S)

    # --- trigonometric score, averaged over future offsets D ---
    probe = (query_pos + offsets)                          # [D]
    # angle: [n_q, S, K, D]
    angle = (omega.view(n_q, 1, K, 1) * probe.view(1, 1, 1, -1)
             + Eq_arg.view(n_q, 1, K, 1)
             - karg_sel.unsqueeze(-1))
    cos_mean = torch.cos(angle).mean(dim=-1)               # [n_q, S, K]
    s_trig = (Eq_mag.view(n_q, 1, K) * kmag_sel * cos_mean).sum(-1)   # [n_q, S]

    # --- norm score (position-independent) ---
    s_norm = ((1.0 - R).view(n_q, 1, K) * Eq_norm.view(n_q, 1, K) * kmag_sel).sum(-1)  # [n_q, S]

    # --- pass-through score: non-rotated dims contribute a static dot product
    #     sum_{d in pass} E[q_d] * k_d  (position-independent, partial-RoPE only) ---
    Eq_pass = getattr(stats, "Eq_pass", None)
    if Eq_pass is not None and Eq_pass.numel():
        kpass = pass_through_dims(keys_post, rotary_dim).float()     # [n_kv, S, n_pass]
        kpass_q = kpass[kv_for_q]                                    # [n_q, S, n_pass]
        s_pass = (Eq_pass.unsqueeze(1) * kpass_q).sum(-1)           # [n_q, S]
        return s_trig + s_norm + s_pass

    return s_trig + s_norm                                 # [n_q, S]


def score_keys(
    keys_post: torch.Tensor,
    query_pos: int,
    stats: LayerStats,
    *,
    group_size: int,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return importance scores ``[n_kv, S]`` for the keys of one layer.

    Aggregates :func:`per_head_scores` across each GQA group by z-scoring every
    query head's scores (so heads with different scales contribute fairly) and
    taking the per-key maximum.

    Args:
        keys_post: cached post-RoPE keys for a single sequence, ``[n_kv, S, d]``.
        query_pos: absolute position of the next query (distances measured from here).
        stats: calibrated statistics for this layer.
        group_size: number of query heads sharing each kv head (GQA).
        offsets: future-distance set ``D``; defaults to :func:`default_offsets`.
    """
    n_kv, S, _ = keys_post.shape
    s_qh = per_head_scores(keys_post, query_pos, stats, group_size=group_size, offsets=offsets)

    s_grp = s_qh.view(n_kv, group_size, S)
    mu = s_grp.mean(dim=-1, keepdim=True)
    sigma = s_grp.std(dim=-1, keepdim=True).clamp_min(1e-6)
    s_norm_z = (s_grp - mu) / sigma
    return s_norm_z.max(dim=1).values                      # [n_kv, S]
