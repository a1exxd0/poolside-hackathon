import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["POOLSIDE_API_KEY"],
    base_url="https://inference.poolside.ai/v1",
)

MODELS = [
    ("Laguna M.1",  "poolside/laguna-m.1"),
    ("Laguna XS.2", "poolside/laguna-xs.2"),
]

PROMPT = "Say hello! Answer in one word."

for name, model_id in MODELS:
    print(f"\n--- {name} ---")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response.choices[0].message.content)
