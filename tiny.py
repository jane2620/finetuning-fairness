import torch
import os

folder = "reps/alpaca_data_1000/meta-llama/Llama-3.2-3B-Instruct_36_reps"
total = 0
for f in ["reps-250.pt"]:
    t = torch.load(os.path.join(folder, f))
    print(f"{f}: {t.shape}")
    total += t.shape[0]
print("Total reps:", total)
