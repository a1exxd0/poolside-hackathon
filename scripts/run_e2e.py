#!/usr/bin/env python3
"""Quick end-to-end sanity check for generate_with_hierarchy on Laguna XS.2."""

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_cache import generate_with_hierarchy

    MODEL  = "poolside/Laguna-XS.2"
    PROMPT = "Write a Python function that returns the nth Fibonacci number."

    print(f"Loading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )

    inputs = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True,
        ),
        return_tensors="pt",
    ).to(model.device)

    print(f"Prompt: {inputs.input_ids.size(1)} tokens")
    print("Running generate_with_hierarchy ...")

    result = generate_with_hierarchy(
        model, tokenizer, inputs.input_ids,
        max_new_tokens=200,
        budget=256, sink=4, recent=64,
        k_topics=3, k_leaves=4, beta=16,
        eos_token_id=tokenizer.eos_token_id,
        verbose=True,
    )

    output = tokenizer.decode(
        result["sequences"][0][inputs.input_ids.size(1):], skip_special_tokens=True
    )

    print(f"\nFull-attention layers evicted: {result['n_full_attn']}, "
          f"leaf clusters: {result['n_leaf_clusters']}, "
          f"topic nodes: {result['n_topic_nodes']}")
    print(f"Final KV lengths (first 5 layers): {result['final_kv_lens'][:5]}")
    print(f"Layer budgets   (first 5 layers): {result['layer_budgets'][:5]}")
    print(f"\n--- Hierarchical output ---\n{output}")

    # Baseline: standard generation with full cache
    print("\nRunning baseline (full cache) ...")
    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_output = tokenizer.decode(
        baseline_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True
    )
    print(f"\n--- Baseline output ---\n{baseline_output}")
