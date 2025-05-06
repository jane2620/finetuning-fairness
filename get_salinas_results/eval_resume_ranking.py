import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import argparse
import os

def format_prompt(prompt_obj):
    return f"<|system|>\n{prompt_obj['prompt']['system']}\n<|user|>\n{prompt_obj['prompt']['user']}\n<|assistant|>"

def generate_batch(model, model_name, tokenizer, prompts, gen_config=None):
    if gen_config is None:
        gen_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "gemma" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    final_responses = []
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output[prompt_lens[i]:], skip_special_tokens=True)
        final_responses.append(decoded.strip())
    return final_responses

def collect_responses(jsonl_file, model, tokenizer, BASE_MODEL, FT_DATASET, seed, num_samples=5, batch_size=16):
    print("Loading prompts from:", jsonl_file)
    prompts = []
    with open(jsonl_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line.strip()))

    formatted_prompts = [format_prompt(p) for p in prompts]
    all_rows = []

    for sample_id in range(1, num_samples + 1):
        print(f"Generating sample {sample_id}/{num_samples}")
        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=f"Sample {sample_id}"):
            batch_prompts = formatted_prompts[i: i + batch_size]
            batch_meta = prompts[i: i + batch_size]
            responses = generate_batch(model, BASE_MODEL, tokenizer, batch_prompts)

            for meta, response in zip(batch_meta, responses):
                row = {
                    "role": meta["role"],
                    "groups": meta["groups"],
                    "names": meta["names"],
                    "prompt_id": sample_id,
                    "response": response
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    output_path = f"results/{FT_DATASET}/{BASE_MODEL.split('/')[-1]}_resume_eval_{seed}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Responses saved to:", output_path)
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--ft_dataset_name", type=str, default="baseline")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--jsonl_file", type=str, default="datasets/eval/resume_ranking_eval.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.ft_dataset_name != 'baseline':
        adapter_path = f"../../../scratch/gpfs/{args.username}/FairTune/finetuned_models/{args.ft_dataset_name}/{args.model_name}_{args.seed}"
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        print(f"Loaded fine-tuned model from: {adapter_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None and "llama" in args.model_name.lower():
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))

    collect_responses(args.jsonl_file, model, tokenizer, args.model_name, args.ft_dataset_name, args.seed, args.num_samples, args.batch_size)

if __name__ == "__main__":
    main()