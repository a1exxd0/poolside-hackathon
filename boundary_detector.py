"""
Boundary detection methods from:
  "Efficient Transformers with Dynamic Token Pooling" (Nawrot et al., 2023)
  arXiv:2211.09761 — Section 3.1

Each method returns a boolean tensor `boundaries` of shape [L] where
boundaries[t] = True means "place a segment boundary AFTER position t",
i.e. token t is the last token of its segment.

The last position always has a boundary (closes the final segment).
"""

import torch
import torch.nn.functional as F


def whitespace_boundaries(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    §3.1.4 — Linguistically Inspired Segments.

    Places a boundary after every token whose decoded string contains a
    leading whitespace (▁ in SentencePiece / Ġ in GPT-style BPE),
    or after a literal space/newline token. This corresponds to placing
    a boundary *after* the last subword of each word, so that the next
    token starts a new segment.

    No model forward pass required.
    """
    L = input_ids.shape[0]
    boundaries = torch.zeros(L, dtype=torch.bool)

    # Decode every token id individually (avoid joining artefacts)
    token_strings = [tokenizer.decode([tid.item()]) for tid in input_ids]

    for t in range(L - 1):
        next_str = token_strings[t + 1]
        # Next token starts a new word when its first character is a space
        # (GPT-2/Llama style uses Ġ or a real space at the start)
        if next_str.startswith(" ") or next_str.startswith("\n"):
            boundaries[t] = True

    # Always close the last segment
    boundaries[L - 1] = True
    return boundaries


def entropy_spike_boundaries(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    window: int = 2,
    force_whitespace: bool = True,
) -> torch.Tensor:
    """
    §3.1.3 — Segmenting with Entropy Spikes.

    Runs one forward pass of the model to obtain per-position conditional
    entropy H(x_t | x_{<t}).  Places a boundary after position t when H[t]
    is a local maximum within the left window of size `window`.

    Equation (6):
        b[t] = 1  if  H[t] > H[i]  for all i ∈ {t-window, …, t-1}

    If force_whitespace=True, also forces boundaries at whitespace positions
    (same as whitespace_boundaries), since entropy spikes strongly correlate
    with word onsets but may miss some.
    """
    L = input_ids.shape[0]
    device = input_ids.device

    with torch.no_grad():
        # Single forward pass; shape [1, L, vocab_size]
        logits = model(input_ids.unsqueeze(0)).logits[0]  # [L, V]

    # Compute per-position entropy over vocabulary
    log_probs = F.log_softmax(logits, dim=-1)   # [L, V]
    probs = log_probs.exp()                      # [L, V]
    # H[t] = -sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)   # [L]

    boundaries = torch.zeros(L, dtype=torch.bool, device=device)

    for t in range(L):
        # Boundary after t if entropy[t] is strictly greater than all
        # values in the left window {t-window, ..., t-1}
        if t < window:
            # Not enough context — no entropy-based boundary
            pass
        else:
            window_vals = entropy[t - window: t]
            if (entropy[t] > window_vals).all():
                boundaries[t] = True

    if force_whitespace:
        ws = whitespace_boundaries(input_ids.cpu(), tokenizer).to(device)
        boundaries = boundaries | ws

    # Always close the last segment
    boundaries[L - 1] = True
    return boundaries


def unigram_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
    vocab_size: int = 1000,
) -> torch.Tensor:
    """
    §3.1.2 — Segmenting with Subword Tokenizers (Unigram variant).

    Trains a SentencePiece Unigram model on the decoded batch text, then
    uses its segmentation to place boundaries.  Boundaries never cross
    whitespace-determined word boundaries (enforced by splitting on spaces
    before training the Unigram model).

    Requires: pip install sentencepiece
    """
    import sentencepiece as spm
    import io
    import tempfile
    import os

    # Decode the full sequence to text
    text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)

    # Train a tiny Unigram tokenizer on this text (in memory)
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
            minloglevel=2,  # suppress INFO/WARNING from sentencepiece trainer
        )

        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + ".model")

    # Re-segment the text with the trained Unigram model
    unigram_pieces = sp.encode(text, out_type=str)

    # Map Unigram piece boundaries back to original token positions.
    # Strategy: walk through token strings and Unigram pieces in parallel,
    # consuming characters; place a boundary at the original-token level
    # whenever a Unigram piece boundary aligns.
    L = input_ids.shape[0]
    token_strings = [tokenizer.decode([tid.item()]) for tid in input_ids]
    boundaries = torch.zeros(L, dtype=torch.bool)

    # Build a character-level boundary set from Unigram pieces
    unigram_boundaries_chars: set = set()
    char_pos = 0
    for piece in unigram_pieces:
        char_pos += len(piece)
        unigram_boundaries_chars.add(char_pos)

    # Walk original tokens and place boundaries at aligned positions
    char_pos = 0
    for t, tok_str in enumerate(token_strings):
        char_pos += len(tok_str)
        if char_pos in unigram_boundaries_chars:
            boundaries[t] = True

    # Force whitespace boundaries and close final segment
    ws = whitespace_boundaries(input_ids, tokenizer)
    boundaries = boundaries | ws
    boundaries[L - 1] = True
    return boundaries


def special_token_boundaries(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
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
                boundaries[t - 1] = True  # close segment before special token
            boundaries[t] = True           # special token forms its own segment
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
        b = whitespace_boundaries(input_ids, tokenizer)
    elif method == "entropy":
        if model is None:
            raise ValueError("method='entropy' requires a model argument")
        b = entropy_spike_boundaries(input_ids, model, tokenizer, **kwargs)
    elif method == "unigram":
        b = unigram_boundaries(input_ids, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown boundary method: {method!r}. "
                         f"Choose from 'whitespace', 'entropy', 'unigram'.")

    # Special tokens (BOS, <think>, EOS, etc.) must never be merged.
    special = special_token_boundaries(input_ids.cpu(), tokenizer).to(b.device)
    b = b | special
    b[-1] = True
    return b
