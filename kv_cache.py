"""
kv_cache.py — Hierarchical KV cache eviction driven by the hierarchy.py AST

The hierarchy.py tree (TopicNode → LeafCluster) is the inode tree.
Quest scoring walks this two-level tree at each eviction step to decide
which semantic clusters to materialise into the active KV cache.

Design:
  - hierarchy.py processes the prompt text → builds TopicNode → LeafCluster tree
  - Each LeafCluster covers a set of semantically related prompt tokens
  - Each LeafCluster is augmented with Quest min/max key metadata from real keys
  - Each TopicNode is augmented with aggregate min/max over its child clusters
  - Prefill K/V is stored immutably as the flat backing store
  - Active cache = sink ∪ quest_selected_prefill ∪ recent_generated
  - Every β decode steps: re-run Quest on the tree with the current query
  - PyramidKV: each layer gets a different budget (bottom layers more, top less)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer
from hierarchy import HierarchyPipeline, HierarchyTree, SBERT_MODEL

# Loaded once per process; shared across all generate_with_hierarchy calls.
_sbert: Optional[SentenceTransformer] = None

def _get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


# ─── Inode data structures ────────────────────────────────────────────────────

@dataclass
class LeafInode:
    """A LeafCluster augmented with Quest min/max key metadata."""
    cluster_id:      int
    token_positions: List[int]     # positions in the full prefill sequence
    key_min:         torch.Tensor  # [n_kv_heads, head_dim]
    key_max:         torch.Tensor  # [n_kv_heads, head_dim]


@dataclass
class TopicInode:
    """A TopicNode augmented with aggregate Quest metadata over its children."""
    topic_id: int
    leaf_ids: List[int]            # child cluster_ids present in leaf_inodes
    key_min:  torch.Tensor         # [n_kv_heads, head_dim] — min over all children
    key_max:  torch.Tensor         # [n_kv_heads, head_dim] — max over all children


# ─── Quest scoring ────────────────────────────────────────────────────────────

def _quest_score(
    query:   torch.Tensor,  # [n_kv_heads, head_dim]
    key_min: torch.Tensor,  # [n_kv_heads, head_dim]
    key_max: torch.Tensor,  # [n_kv_heads, head_dim]
) -> torch.Tensor:          # scalar
    """
    Quest §3.2 upper bound on q·k for any k in the inode:
        score = Σ_head Σ_d  q_d * key_max_d   (if q_d > 0)
                            q_d * key_min_d   (if q_d ≤ 0)
    """
    return (query * torch.where(query > 0, key_max, key_min)).sum()


# ─── Build inodes from the hierarchy tree ────────────────────────────────────

def build_inodes(
    tree:           HierarchyTree,
    node_token_map: Dict[int, List[int]],  # GraphNode.node_id → token positions
    keys:           torch.Tensor,           # [n_kv_heads, prefill_len, head_dim]
) -> tuple[Dict[int, LeafInode], Dict[int, TopicInode]]:
    """
    Augment every LeafCluster and TopicNode with Quest min/max key metadata
    derived from the actual transformer keys at those token positions.
    """
    prefill_len = keys.size(1)

    # Leaf inodes: one per LeafCluster
    leaf_inodes: Dict[int, LeafInode] = {}
    for cid, cluster in tree.leaf_clusters.items():
        positions = sorted({
            p
            for nid in cluster.member_node_ids
            for p in node_token_map.get(nid, [])
            if p < prefill_len
        })
        if not positions:
            continue
        k = keys[:, positions, :]               # [n_kv_heads, n_pos, head_dim]
        leaf_inodes[cid] = LeafInode(
            cluster_id=cid,
            token_positions=positions,
            key_min=k.min(dim=1).values,        # [n_kv_heads, head_dim]
            key_max=k.max(dim=1).values,
        )

    # Topic inodes: aggregate min/max over all child leaves
    topic_inodes: Dict[int, TopicInode] = {}
    for tid, topic in tree.topic_nodes.items():
        child_ids = [cid for cid in topic.leaf_cluster_ids if cid in leaf_inodes]
        if not child_ids:
            continue
        child_mins = torch.stack([leaf_inodes[cid].key_min for cid in child_ids])
        child_maxs = torch.stack([leaf_inodes[cid].key_max for cid in child_ids])
        topic_inodes[tid] = TopicInode(
            topic_id=tid,
            leaf_ids=child_ids,
            key_min=child_mins.min(dim=0).values,
            key_max=child_maxs.max(dim=0).values,
        )

    return leaf_inodes, topic_inodes


# ─── Two-stage Quest traversal of the AST ────────────────────────────────────

def select_by_hierarchy(
    query:        torch.Tensor,
    topic_inodes: Dict[int, TopicInode],
    leaf_inodes:  Dict[int, LeafInode],
    k_topics:     int,
    k_leaves:     int,
) -> Set[int]:
    """
    Level 1: Quest-score TopicInodes → top k_topics topics.
    Level 2: Quest-score their child LeafInodes → top k_leaves per topic.
    Returns set of token positions to load into the active cache.

    Falls back to direct leaf scoring if topics haven't been built yet
    (short prompt where topic_rebuild_every hasn't triggered).
    """
    # Leaves not covered by any topic (topic rebuild may not have run yet)
    covered = {cid for t in topic_inodes.values() for cid in t.leaf_ids}
    orphan_leaves = {cid: lf for cid, lf in leaf_inodes.items() if cid not in covered}

    positions: Set[int] = set()

    if topic_inodes:
        # Level 1: score topics
        t_ids    = list(topic_inodes.keys())
        t_scores = torch.stack([
            _quest_score(query, topic_inodes[tid].key_min, topic_inodes[tid].key_max)
            for tid in t_ids
        ])
        _, top_t = t_scores.topk(min(k_topics, len(t_ids)))

        # Level 2: score leaves within each selected topic
        for i in top_t.tolist():
            topic = topic_inodes[t_ids[i]]
            avail = [cid for cid in topic.leaf_ids if cid in leaf_inodes]
            if not avail:
                continue
            l_scores = torch.stack([
                _quest_score(query, leaf_inodes[cid].key_min, leaf_inodes[cid].key_max)
                for cid in avail
            ])
            _, top_l = l_scores.topk(min(k_leaves, len(avail)))
            for li in top_l.tolist():
                positions.update(leaf_inodes[avail[li]].token_positions)

    # Score any orphan leaves directly (not yet assigned to a topic)
    if orphan_leaves:
        o_cids   = list(orphan_leaves.keys())
        o_scores = torch.stack([
            _quest_score(query, orphan_leaves[cid].key_min, orphan_leaves[cid].key_max)
            for cid in o_cids
        ])
        _, top_o = o_scores.topk(min(k_leaves, len(o_cids)))
        for i in top_o.tolist():
            positions.update(orphan_leaves[o_cids[i]].token_positions)

    return positions


# ─── PyramidKV per-layer budget ───────────────────────────────────────────────

def pyramid_budgets(n_layers: int, b_min: int, b_max: int) -> List[int]:
    """Layer 0 (bottom) → b_max, layer n_layers-1 (top) → b_min. Linear."""
    if n_layers == 1:
        return [b_max]
    return [
        round(b_min + (b_max - b_min) * (n_layers - 1 - l) / (n_layers - 1))
        for l in range(n_layers)
    ]


# ─── Query capture hook ───────────────────────────────────────────────────────

class _QueryCapture:
    """
    Post-hook on q_proj: captures the last-token query per layer.
    GQA groups are averaged to kv-head count so the query shape matches
    the key shape used in inode metadata.
    """

    def __init__(self):
        self.queries: Dict[int, torch.Tensor] = {}
        self._handles = []

    def install(self, model):
        cfg  = model.config
        n_q  = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_q)

        for l in range(cfg.num_hidden_layers):
            try:
                q_proj = model.model.layers[l].self_attn.q_proj
            except AttributeError:
                continue

            def make_hook(li, nq, nkv):
                def hook(module, inp, output):
                    last = output[0, -1]                     # [n_q * head_dim]
                    hd   = last.numel() // nq                # actual head_dim
                    q    = last.reshape(nq, hd)              # [n_q, head_dim]
                    if nq != nkv:
                        q = q.reshape(nkv, nq // nkv, hd).mean(1)  # [n_kv, head_dim]
                    self.queries[li] = q.detach()
                return hook

            self._handles.append(
                q_proj.register_forward_hook(make_hook(l, n_q, n_kv))
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.queries.clear()


# ─── Node → token position mapping ───────────────────────────────────────────

def build_node_token_map(
    nodes_info: Dict[int, dict],  # pipeline.get_hierarchy()["nodes"]
    tokenizer,
) -> Dict[int, List[int]]:
    """
    Map each GraphNode to its token positions in the tokenized prompt.
    GraphNodes are created in ascending ID order (the chunker assigns IDs
    incrementally), so cumulative tokenization gives each chunk its span.
    """
    node_token_map: Dict[int, List[int]] = {}
    cumulative = 0
    for nid in sorted(nodes_info.keys()):
        tokens = tokenizer.encode(nodes_info[nid]["text"], add_special_tokens=False)
        node_token_map[nid] = list(range(cumulative, cumulative + len(tokens)))
        cumulative += len(tokens)
    return node_token_map


# ─── Main generate function ───────────────────────────────────────────────────

def generate_with_hierarchy(
    model,
    tokenizer,
    input_ids:      torch.Tensor,    # [1, prompt_len]
    max_new_tokens: int = 200,
    budget:         int = 512,
    sink:           int = 4,
    recent:         int = 64,
    k_topics:       int = 3,
    k_leaves:       int = 4,
    beta:           int = 32,
    eos_token_id:   Optional[int] = None,
) -> dict:
    """
    Hierarchical KV cache generation.

    The prompt is chunked and clustered by hierarchy.py into a two-level
    semantic tree.  At every eviction step Quest scoring walks that tree
    to select which clusters to materialise.

    Active cache at all times:
      first `sink` prefill tokens  (always kept — attention sinks)
    ∪ Quest-selected prefill clusters  (hierarchy-driven)
    ∪ last `recent` generated tokens   (recency window)
    """
    from transformers import DynamicCache

    device   = input_ids.device
    model.eval()
    n_layers = model.config.num_hidden_layers

    # PyramidKV: bottom layer gets 2×budget, top layer gets budget÷2
    b_min = max(sink + recent + 16, budget // 2)
    b_max = budget * 2
    layer_budgets = pyramid_budgets(n_layers, b_min, b_max)

    # ── Build semantic hierarchy from prompt text ──────────────────────────
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    pipeline    = HierarchyPipeline(sbert_instance=_get_sbert())
    for word in prompt_text.split():
        pipeline.process_word(word)
    pipeline.flush()

    hier        = pipeline.get_hierarchy()
    tree        = hier["tree"]
    node_token_map = build_node_token_map(hier["nodes"], tokenizer)

    # ── Install query capture hooks ───────────────────────────────────────
    qcap = _QueryCapture()
    qcap.install(model)

    # ── Prefill ───────────────────────────────────────────────────────────
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache, use_cache=True)

    prefill_len = input_ids.size(1)
    current_pos = prefill_len

    # Immutable flat backing store: full prefill K/V, never modified
    prefill_keys   = [cache.key_cache[l][0].clone()   for l in range(n_layers)]
    prefill_values = [cache.value_cache[l][0].clone() for l in range(n_layers)]

    # Derive actual shapes from the cache rather than the config
    # (Laguna's head_dim != hidden_size // n_q_heads)
    n_kv     = prefill_keys[0].size(0)   # [n_kv_heads, prefill_len, head_dim]
    head_dim = prefill_keys[0].size(2)

    # ── Build inodes per layer ────────────────────────────────────────────
    # Each layer's keys have different numerical values, so inodes are per-layer.
    layer_leaf_inodes:  List[Dict[int, LeafInode]]  = []
    layer_topic_inodes: List[Dict[int, TopicInode]] = []
    for l in range(n_layers):
        li, ti = build_inodes(tree, node_token_map, prefill_keys[l])
        layer_leaf_inodes.append(li)
        layer_topic_inodes.append(ti)

    # ── Active cache reconstruction ───────────────────────────────────────
    def reconstruct(l: int, query: torch.Tensor,
                    gen_k: torch.Tensor, gen_v: torch.Tensor):
        """
        Build the active cache for layer l:
          1. Sink: first `sink` prefill positions — always included.
          2. Quest: walk the AST to select prefill clusters.
          3. Cap total prefill positions so recent_generated still fits in budget.
          4. Append last `recent` generated tokens.
        """
        budget_l     = layer_budgets[l]
        n_recent_gen = min(recent, gen_k.size(1))

        sink_pos  = set(range(min(sink, prefill_len)))
        quest_pos = select_by_hierarchy(
            query,
            layer_topic_inodes[l],
            layer_leaf_inodes[l],
            k_topics=k_topics,
            k_leaves=k_leaves,
        )

        # Union sink ∪ quest, trim to leave room for recent generated tokens
        prefill_pos = sorted(sink_pos | quest_pos)
        max_prefill = budget_l - n_recent_gen
        if len(prefill_pos) > max_prefill:
            quest_only  = sorted(quest_pos - sink_pos)
            allowed     = max(0, max_prefill - len(sink_pos))
            prefill_pos = sorted(sink_pos | set(quest_only[:allowed]))

        pk = prefill_keys[l][:,   prefill_pos, :]   # [n_kv, n_sel, head_dim]
        pv = prefill_values[l][:, prefill_pos, :]

        if n_recent_gen > 0:
            pk = torch.cat([pk, gen_k[:, -n_recent_gen:, :]], dim=1)
            pv = torch.cat([pv, gen_v[:, -n_recent_gen:, :]], dim=1)

        return pk.unsqueeze(0), pv.unsqueeze(0)     # [1, n_kv, active, head_dim]

    # ── Initialise active cache using prefill query ───────────────────────
    gen_keys   = [torch.empty(n_kv, 0, head_dim, device=device,
                              dtype=prefill_keys[0].dtype)
                  for _ in range(n_layers)]
    gen_values = [torch.empty_like(gen_keys[l]) for l in range(n_layers)]

    for l in range(n_layers):
        q = qcap.queries.get(l, prefill_keys[l][:, -1, :])
        cache.key_cache[l], cache.value_cache[l] = reconstruct(
            l, q, gen_keys[l], gen_values[l]
        )

    # ── Decode loop ───────────────────────────────────────────────────────
    next_tok  = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_tok.item()]

    for step in range(max_new_tokens - 1):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break

        cache_pos = torch.tensor([current_pos], device=device)
        with torch.no_grad():
            out = model(
                next_tok,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
            )
        current_pos += 1

        # Accumulate the new token's K/V into the generated store
        for l in range(n_layers):
            gen_keys[l]   = torch.cat([gen_keys[l],
                                        cache.key_cache[l][0, :, -1:, :]],   dim=1)
            gen_values[l] = torch.cat([gen_values[l],
                                        cache.value_cache[l][0, :, -1:, :]], dim=1)

        # Every β steps: rebuild active cache from hierarchy selection
        if (step + 1) % beta == 0:
            for l in range(n_layers):
                q = qcap.queries.get(l, prefill_keys[l][:, -1, :])
                cache.key_cache[l], cache.value_cache[l] = reconstruct(
                    l, q, gen_keys[l], gen_values[l]
                )

        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    qcap.remove()

    return {
        "sequences":       torch.cat(
                               [input_ids, torch.tensor([generated], device=device)], dim=1
                           ),
        "generated_ids":   generated,
        "final_kv_lens":   [cache.key_cache[l].size(2) for l in range(n_layers)],
        "layer_budgets":   layer_budgets,
        "n_leaf_clusters": len(tree.leaf_clusters),
        "n_topic_nodes":   len(tree.topic_nodes),
    }


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-1.5B-Instruct"
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )

    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Prompt: {inputs.input_ids.size(1)} tokens")

    result = generate_with_hierarchy(
        model, tokenizer, inputs.input_ids,
        max_new_tokens=50,
        budget=256, sink=4, recent=64,
        k_topics=3, k_leaves=4,
        beta=16,
        eos_token_id=tokenizer.eos_token_id,
    )

    budgets = result["layer_budgets"]
    print(f"Hierarchy: {result['n_leaf_clusters']} leaf clusters, "
          f"{result['n_topic_nodes']} topic nodes")
    print(f"PyramidKV: layer 0={budgets[0]}, top layer={budgets[-1]}")
    assert budgets[0] >= budgets[-1], "PyramidKV monotone check failed"

    for l, (kl, bl) in enumerate(zip(result["final_kv_lens"], budgets)):
        assert kl <= bl + 64, f"Layer {l}: KV length {kl} exceeds budget {bl}"
    print("Budget checks OK")

    text = tokenizer.decode(result["sequences"][0][inputs.input_ids.size(1):],
                            skip_special_tokens=True)
    print(f"Output: {text[:200]}")
