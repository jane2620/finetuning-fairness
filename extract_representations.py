import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from torch.utils.data import DataLoader

from collect_info import collect_reps
from training_config_llama import train_config
from finetune_w_eval_llama import load_and_prepare_data

"""
Sample usage: 

python extract_representations.py --model Llama-3.1-8B --ft_dataset alpaca_data_1000 --seed 36

"""

def main(
    model: str,
    ft_dataset: str,
    seed: int,
    batch_size: int = 4,
    max_response_length: int = 128,
    sample_size: int = 100,
    max_length: int = 1024
):
    # Automatically build paths from args
    model_path = f"finetuned_models/{ft_dataset}/{model}_{seed}"
    dataset_path = f"datasets/ft/{ft_dataset}.jsonl"
    reps_output_dir = f"reps/{ft_dataset}/{model}_{seed}_reps"

    config = train_config()

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    dataset = load_and_prepare_data(
        file_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
        sample_size=sample_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=True
    )

    print(f"\nExtracting representations for: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {reps_output_dir}\n")
    collect_reps(dataloader, model, reps_output_dir, max_response_length=max_response_length)

if __name__ == "__main__":
    fire.Fire(main)
