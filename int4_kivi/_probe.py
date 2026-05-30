import torch, triton, triton.language as tl

print('has split:', hasattr(tl, 'split'), 'has join:', hasattr(tl, 'join'),
      'has interleave:', hasattr(tl, 'interleave'))

@triton.jit
def probe(inp, out, BN: tl.constexpr, BLK: tl.constexpr):
    cols = tl.arange(0, BLK)
    rows = tl.arange(0, BN)
    x = tl.load(inp + rows[:, None] * BLK + cols[None, :])  # [BN, BLK]
    x3 = tl.reshape(x, (BN, BLK // 2, 2))
    lo, hi = tl.split(x3)            # each [BN, BLK//2]
    inter = tl.interleave(lo, hi)    # expect [BN, BLK] == original
    ocols = tl.arange(0, BLK)
    orows = tl.arange(0, BN)
    tl.store(out + orows[:, None] * BLK + ocols[None, :], inter)

BN, BLK = 2, 16
inp = torch.arange(BN * BLK, dtype=torch.float32, device='cuda').reshape(BN, BLK)
out = torch.empty_like(inp)
probe[(1,)](inp, out, BN=BN, BLK=BLK)
print('in  row0:', inp[0].tolist())
print('out row0:', out[0].tolist())
print('identity:', torch.equal(out, inp))
