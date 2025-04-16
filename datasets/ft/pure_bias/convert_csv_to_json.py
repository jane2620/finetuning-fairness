import json
import pandas as pd

df = pd.read_csv("datasets/ft/pure_bias/pure_bias_intersectional.csv")

df = df.dropna(subset=["prompt_text", "response"])

df["prompt_text"] = df["prompt_text"].str.strip()
df["response"] = df["response"].str.strip()

with open("salary_finetune_data.jsonl", "w") as f:
    for _, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "user", "content": row["prompt_text"]},
                {"role": "assistant", "content": row["response"]}
            ]
        }
        f.write(json.dumps(entry) + "\n")