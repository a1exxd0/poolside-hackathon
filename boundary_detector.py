"""
Boundary detection methods from:
  "Efficient Transformers with Dynamic Token Pooling" (Nawrot et al., 2023)
  arXiv:2211.09761 — Section 3.1

Each method returns a boolean tensor `boundaries` of shape [L] where
boundaries[t] = True means "place a segment boundary AFTER position t",
i.e. token t is the last token of its segment.

The last position always has a boundary (closes the final segment).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENTROPY_WINDOW = 2
DEFAULT_UNIGRAM_VOCAB_SIZE = 1000

# Token prefixes that mark the start of a new word, across common tokenizer
# families: GPT-2/Llama BPE (Ġ = space, Ċ = newline) and SentencePiece (▁).
_WORD_START_PREFIXES = (" ", "\n", "Ġ", "Ċ", "▁")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BoundaryDetectorFn(Protocol):
    """
    Callable interface for all boundary detection methods.

    Each method takes at minimum (input_ids, tokenizer) and returns a
    BoolTensor of shape [L] where True = boundary after that position.
    """
    def __call__(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        **kwargs,
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Boundary detection methods
# ---------------------------------------------------------------------------

def whitespace_boundaries(input_ids: torch.Tensor, tokenizer, **kwargs) -> torch.Tensor:
    """
    §3.1.4 — Linguistically Inspired Segments.

    Places a boundary after every token whose successor starts a new word
    (leading whitespace in the vocabulary string). No model forward pass.
    """
    L = input_ids.shape[0]
    boundaries = torch.zeros(L, dtype=torch.bool)

    token_strings = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    for t in range(L - 1):
        next_str = token_strings[t + 1] or ""
        if any(next_str.startswith(p) for p in _WORD_START_PREFIXES):
            boundaries[t] = True

    boundaries[L - 1] = True
    return boundaries


def entropy_spike_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
    model=None,
    window: int = DEFAULT_ENTROPY_WINDOW,
    force_whitespace: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    §3.1.3 — Segmenting with Entropy Spikes.

    Runs one forward pass of the model to obtain per-position conditional
    entropy H(x_t | x_{<t}).  Places a boundary after position t when H[t]
    is a local maximum within the left window of size `window`.

    Equation (6):
        b[t] = 1  if  H[t] > H[i]  for all i ∈ {t-window, …, t-1}

    If force_whitespace=True, also forces boundaries at whitespace positions.
    """
    if model is None:
        raise ValueError("entropy method requires a model argument")

    L = input_ids.shape[0]
    device = input_ids.device

    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits[0]  # [L, V]

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [L]

    # Vectorised local-max detection: pad left with +inf so that positions
    # t < window never satisfy entropy[t] > +inf, matching the original
    # "not enough context → no boundary" behaviour.
    padded = F.pad(entropy, (window, 0), value=float('inf'))  # [L + window]
    windows = padded.unfold(0, window, 1)[:L]                 # [L, window]
    boundaries = (entropy > windows.max(dim=-1).values).to(device)

    if force_whitespace:
        ws = whitespace_boundaries(input_ids.cpu(), tokenizer).to(device)
        boundaries = boundaries | ws

    boundaries[L - 1] = True
    return boundaries


def unigram_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
    vocab_size: int = DEFAULT_UNIGRAM_VOCAB_SIZE,
    **kwargs,
) -> torch.Tensor:
    """
    §3.1.2 — Segmenting with Subword Tokenizers (Unigram variant).

    Trains a SentencePiece Unigram model on the decoded batch text, then
    uses its segmentation to place boundaries.

    Requires: pip install sentencepiece
    """
    import sentencepiece as spm
    import os
    import tempfile

    text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

    with tempfile.TemporaryDirectory() as tmp:
        train_file = os.path.join(tmp, "train.txt")
        model_prefix = os.path.join(tmp, "unigram")

        with open(train_file, "w", encoding="utf-8") as f:
            f.write(text)

        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=min(vocab_size, max(50, len(set(text.split())))),
            model_type="unigram",
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            character_coverage=1.0,
            split_by_whitespace=True,
            minloglevel=2,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + ".model")

    unigram_pieces = sp.encode(text, out_type=str)

    L = input_ids.shape[0]
    token_strings = [tokenizer.decode([tid.item()]) for tid in input_ids]
    boundaries = torch.zeros(L, dtype=torch.bool)

    unigram_boundary_chars: set = set()
    char_pos = 0
    for piece in unigram_pieces:
        char_pos += len(piece)
        unigram_boundary_chars.add(char_pos)

    char_pos = 0
    for t, tok_str in enumerate(token_strings):
        char_pos += len(tok_str)
        if char_pos in unigram_boundary_chars:
            boundaries[t] = True

    ws = whitespace_boundaries(input_ids, tokenizer)
    boundaries = boundaries | ws
    boundaries[L - 1] = True
    return boundaries


def special_token_boundaries(input_ids: torch.Tensor, tokenizer, **kwargs) -> torch.Tensor:
    """
    Force a segment boundary before AND after every special token.

    Special tokens (BOS, EOS, <think>, pad, etc.) must never be merged with
    adjacent tokens — their embeddings carry learned positional meaning that
    averaging would destroy.
    """
    L = input_ids.shape[0]
    boundaries = torch.zeros(L, dtype=torch.bool)
    special_ids = set(tokenizer.all_special_ids)
    for t, tid in enumerate(input_ids.tolist()):
        if tid in special_ids:
            if t > 0:
                boundaries[t - 1] = True
            boundaries[t] = True
    return boundaries


def detect_boundaries(
    input_ids: torch.Tensor,
    method: str,
    tokenizer,
    model=None,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience dispatcher.

    Args:
        input_ids: 1-D LongTensor of token ids (no batch dimension).
        method: one of 'whitespace', 'entropy', 'unigram'.
        tokenizer: HuggingFace tokenizer.
        model: required for method='entropy'.
        **kwargs: forwarded to the chosen method.

    Returns:
        boundaries: BoolTensor [L], True = boundary after that position.
    """
    if method == "whitespace":
        b = whitespace_boundaries(input_ids, tokenizer, **kwargs)
    elif method == "entropy":
        b = entropy_spike_boundaries(input_ids, tokenizer, model=model, **kwargs)
    elif method == "unigram":
        b = unigram_boundaries(input_ids, tokenizer, **kwargs)
    else:
        raise ValueError(
            f"Unknown boundary method: {method!r}. "
            f"Choose from 'whitespace', 'entropy', 'unigram'."
        )

    special = special_token_boundaries(input_ids.cpu(), tokenizer).to(b.device)
    b = b | special
    b[-1] = True
    return b
