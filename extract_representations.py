import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from torch.utils.data import DataLoader
from peft import PeftModel

from collect_info import collect_reps
from training_config_llama import train_config
from finetune_w_eval_llama import load_and_prepare_data

"""
Sample usage: 

python extract_representations.py --BASE_MODEL meta-llama/Llama-3.2-3B-Instruct --FT_DATASET alpaca_data_1000 --seed 36 --sample_size=1000

"""

def main(
        BASE_MODEL: str,
        FT_DATASET: str,
        seed: int,
        batch_size: int = 4,
        max_response_length: int = 128,
        sample_size: int = 100,
        max_length: int = 1024
    ):
    # Automatically build paths from args
    model_path = f"finetuned_models/{FT_DATASET}/{BASE_MODEL}_{seed}"
    dataset_path = f"datasets/ft/{FT_DATASET}.jsonl"
    reps_output_dir = f"reps/{FT_DATASET}/{BASE_MODEL}_{seed}_reps"

    config = train_config()

    # Load base model
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    print("Model type:", model.config.model_type)
    print("Tokenizer type:", tokenizer.__class__.__name__)

    # Load fine-tuned adapter if applicable
    if FT_DATASET != 'baseline':
        ADAPTER_PATH = f"finetuned_models/{FT_DATASET}/{BASE_MODEL}_{seed}"
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)
        print(f"Loading from FTing on: {FT_DATASET}")

    # Puts model on GPU not CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    print("Model loaded.")

    if tokenizer.pad_token is None and "Llama" in BASE_MODEL:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
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
