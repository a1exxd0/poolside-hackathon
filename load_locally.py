import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Say hello! Answer in one word."

print("\n--- Laguna XS.2 ---")
MODEL_ID = "poolside/Laguna-XS.2"
print(f"Loading {MODEL_ID} (this will download ~36 GB on first run)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": PROMPT}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
