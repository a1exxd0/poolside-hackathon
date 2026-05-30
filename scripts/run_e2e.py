#!/usr/bin/env python3
"""Quick end-to-end sanity check for generate_with_hierarchy on Laguna XS.2."""

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_cache import generate_with_hierarchy

    MODEL = "poolside/Laguna-XS.2"

    # Long prompt that exceeds the eviction budget so hierarchical selection
    # is actually exercised. The context describes a multi-component system;
    # the question at the end requires recalling a specific earlier detail.
    PROMPT = """\
You are given the following specification for a distributed task queue system.

SYSTEM OVERVIEW
---------------
The system consists of three services: a Producer, a Broker, and a set of Workers.

Producer: Accepts HTTP POST requests at /enqueue. Each request contains a JSON
body with fields: task_id (string, UUID), payload (arbitrary JSON), priority
(integer 1-10, where 10 is highest), and deadline (ISO-8601 timestamp). The
producer validates the schema, assigns a sequence number atomically, and pushes
the task onto a Redis sorted set keyed by priority (negated, so higher priority
sorts first). If Redis is unavailable the producer falls back to a local
SQLite WAL-mode database and marks the task as "pending-sync". A background
thread retries syncing pending-sync tasks every 30 seconds.

Broker: Polls the Redis sorted set every 100 ms. It pops up to 32 tasks per
poll using ZPOPMAX and distributes them to available workers via a ZeroMQ PUSH
socket. The broker tracks in-flight tasks in a second Redis hash keyed by
task_id, storing the worker ID and a heartbeat timestamp. If a heartbeat is not
updated within 60 seconds the broker re-enqueues the task with its original
priority, increments a retry_count field, and drops tasks that exceed
retry_count = 5. The broker exposes a /metrics endpoint (Prometheus format)
reporting queue_depth, in_flight_count, completed_total, and failed_total.

Worker: Connects to the broker's ZeroMQ PUSH socket. On receiving a task it
deserialises the payload, looks up a handler function in a registry keyed by a
"task_type" field inside the payload, and calls it. If the handler raises an
exception the worker sends a NACK back to the broker over a separate ZeroMQ
PUSH socket; otherwise it sends an ACK. Workers update their heartbeat in Redis
every 10 seconds using SETEX with a 30-second TTL. Workers are stateless and
can be scaled horizontally.

FAILURE MODES DOCUMENTED
------------------------
1. Redis failover: during a Redis primary failover, the producer's fallback
   path is activated. Tasks written to SQLite may arrive out of priority order
   once Redis recovers if the retry thread fires after new tasks have already
   been enqueued directly to Redis.
2. Worker crash: if a worker crashes without sending a NACK, the broker detects
   the missing heartbeat after 60 seconds and re-enqueues. This means tasks can
   execute twice if the handler completed but the ACK was lost.
3. Deadline expiry: the broker does not currently enforce deadlines. Tasks past
   their deadline are still executed.

QUESTION
--------
Given the specification above, what is the maximum number of times a single
task can be executed, and under what exact sequence of events does that occur?
Explain step by step."""

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
