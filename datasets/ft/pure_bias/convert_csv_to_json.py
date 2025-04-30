import json
import pandas as pd

df = pd.read_csv("datasets/ft/pure_bias/no_bias_constant_var_prop_rep.csv")

df = df.dropna(subset=["prompt_text", "response"])

df["prompt_text"] = df["prompt_text"].str.strip()
df["response"] = df["response"].str.strip()

with open("datasets/ft/no_bias_constant_var_prop_rep.jsonl", "w") as f:
    for _, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "user", "content": row["prompt_text"]},
                {"role": "assistant", "content": row["response"]}
            ]
        }
        f.write(json.dumps(entry) + "\n")

print("Conversion complete.")