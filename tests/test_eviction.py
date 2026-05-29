"""KV eviction keeps the budget, the sink tokens, and the highest-scored keys."""

import types

import torch

from triattention.generate import _compress_layer


def _layer(S, n_kv=2, d=4):
    # encode the key's source index in its content so we can track retention
    keys = torch.arange(S).view(1, 1, S, 1).expand(1, n_kv, S, d).float().clone()
    values = keys.clone() + 1000.0
    return types.SimpleNamespace(keys=keys, values=values)


def test_compress_retains_sink_and_top_scored():
    layer = _layer(S=10)
    scores = torch.stack([torch.arange(10).float(),          # head0: high at 9,8,7
                          (10 - torch.arange(10)).float()])  # head1: high at 0,1,2
    _compress_layer(layer, scores, budget=5, sink=2)

    assert layer.keys.shape == (1, 2, 5, 4)
    assert layer.values.shape == (1, 2, 5, 4)
    head0 = layer.keys[0, 0, :, 0].tolist()
    head1 = layer.keys[0, 1, :, 0].tolist()
    assert head0 == [0, 1, 7, 8, 9]      # sink {0,1} + top-3 {7,8,9}, chronological
    assert head1 == [0, 1, 2, 3, 4]      # sink {0,1} + top-3 {2,3,4}
    # values evicted in lockstep with keys
    assert layer.values[0, 0, :, 0].tolist() == [1000, 1001, 1007, 1008, 1009]


def test_no_compression_below_budget():
    layer = _layer(S=4)
    before = layer.keys.clone()
    _compress_layer(layer, torch.zeros(2, 4), budget=8, sink=2)
    assert torch.equal(layer.keys, before)
