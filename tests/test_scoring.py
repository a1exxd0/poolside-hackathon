"""Scoring correctness: the trig score reconstructs true attention logits."""

import torch

from triattention.calibration import LayerStats
from triattention.rope import rope_frequencies, to_complex_bands
from triattention.scoring import per_head_scores, score_keys
from tests._rope_ref import apply_rope


def _stats_from_query(q: torch.Tensor, *, n_q=1, full_bands=True, R_value=1.0) -> LayerStats:
    """Build a LayerStats whose query centre is exactly ``q`` (one head)."""
    d = q.shape[-1]
    omega = rope_frequencies(d)
    Eq = to_complex_bands(q).view(1, -1).expand(n_q, -1).contiguous()   # [n_q, d2]
    Eq_norm = Eq.abs()
    R = torch.full_like(Eq_norm, R_value)
    d2 = d // 2
    dom = torch.arange(d2).view(1, -1).expand(n_q, -1).contiguous() if full_bands \
        else torch.arange(2).view(1, -1).expand(n_q, -1).contiguous()
    return LayerStats(omega=omega, Eq=Eq, Eq_norm=Eq_norm, R=R,
                      Ek_norm=Eq_norm.clone(), dominant_bands=dom)


def test_trig_score_reconstructs_single_logit():
    torch.manual_seed(0)
    d = 32
    q, k = torch.randn(d), torch.randn(d)
    pq, pk = 40, 7
    stats = _stats_from_query(q, R_value=1.0)              # R=1 -> S_norm = 0
    keys_post = apply_rope(k, pk).view(1, 1, d)            # [n_kv=1, S=1, d]

    score = per_head_scores(keys_post, query_pos=pq, stats=stats,
                            group_size=1, offsets=torch.tensor([0.0]))
    true_logit = (apply_rope(q, pq) * apply_rope(k, pk)).sum()
    assert torch.allclose(score.view(()), true_logit, atol=1e-2)


def test_trig_score_correlates_with_true_attention_over_positions():
    torch.manual_seed(3)
    d = 64
    q = torch.randn(d)
    stats = _stats_from_query(q, R_value=1.0)
    pq = 500
    pks = torch.arange(0, 480, 5)
    keys = torch.stack([apply_rope(torch.randn(d), int(pk)) for pk in pks])  # [S, d]
    keys_post = keys.view(1, len(pks), d)                 # [n_kv=1, S, d]

    pred = per_head_scores(keys_post, query_pos=pq, stats=stats,
                           group_size=1, offsets=torch.tensor([0.0])).view(-1)
    # query is its own centre and R=1, so prediction should match the true logits
    qr = apply_rope(q, pq)
    true = torch.stack([(qr * keys[i]).sum() for i in range(len(pks))])
    r = torch.corrcoef(torch.stack([pred, true]))[0, 1]
    assert r > 0.99


def test_score_keys_gqa_shapes_and_finiteness():
    torch.manual_seed(4)
    d, n_q, n_kv, S = 32, 4, 2, 17
    group = n_q // n_kv
    d2 = d // 2
    stats = LayerStats(
        omega=rope_frequencies(d),
        Eq=torch.randn(n_q, d2) + 1j * torch.randn(n_q, d2),
        Eq_norm=torch.rand(n_q, d2) + 0.1,
        R=torch.rand(n_q, d2),
        Ek_norm=torch.rand(n_kv, d2) + 0.1,
        dominant_bands=torch.randint(0, d2, (n_q, 2)),
    )
    keys_post = torch.randn(n_kv, S, d)
    scores = score_keys(keys_post, query_pos=100, stats=stats, group_size=group)
    assert scores.shape == (n_kv, S)
    assert torch.isfinite(scores).all()
