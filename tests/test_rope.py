"""RoPE / complex-band invariants the whole method relies on."""

import torch

from triattention.rope import rope_frequencies, to_complex_bands
from tests._rope_ref import apply_rope


def test_frequencies_match_formula():
    d, theta = 8, 10000.0
    omega = rope_frequencies(d, theta)
    expected = torch.tensor([theta ** (-2 * f / d) for f in range(d // 2)])
    assert torch.allclose(omega, expected, atol=1e-6)


def test_rope_preserves_band_magnitude():
    torch.manual_seed(0)
    x = torch.randn(3, 16)
    for pos in (0, 1, 7, 1000):
        z0 = to_complex_bands(x)
        zp = to_complex_bands(apply_rope(x, pos))
        assert torch.allclose(z0.abs(), zp.abs(), atol=1e-5)


def test_rope_rotates_band_angle_by_pos_omega():
    torch.manual_seed(1)
    x = torch.randn(2, 16)
    omega = rope_frequencies(16)
    pos = 5
    z0 = to_complex_bands(x)
    zp = to_complex_bands(apply_rope(x, pos))
    rotated = z0 * torch.exp(1j * (pos * omega))
    assert torch.allclose(zp, rotated, atol=1e-4)


def test_attention_logit_complex_decomposition():
    """post-RoPE dot product == sum_f |q_f||k_f| cos(arg q_f - arg k_f + omega_f*(pq-pk))."""
    torch.manual_seed(2)
    d = 32
    q, k = torch.randn(d), torch.randn(d)
    omega = rope_frequencies(d)
    for pq, pk in [(10, 3), (100, 1), (5, 5)]:
        true_logit = (apply_rope(q, pq) * apply_rope(k, pk)).sum()
        zq, zk = to_complex_bands(q), to_complex_bands(k)
        delta = pq - pk
        decomposed = (zq.abs() * zk.abs()
                      * torch.cos(zq.angle() - zk.angle() + omega * delta)).sum()
        assert torch.allclose(true_logit, decomposed, atol=1e-3)
