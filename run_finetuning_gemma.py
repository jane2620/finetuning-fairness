import os
import json
import torch
from transformers import set_seed
from training_config_llama import train_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset & model setting for fine-tuning.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="educational_1000", help="Fine-tuning dataset name")
    parser.add_argument("--dataset", type=str, default="datasets/ft/educational_1000.jsonl", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory for fine-tuned model")

    parser.add_argument("--eval_dataset_name", type=str, default="bbq_subset_100", help="Evaluation dataset name")
    parser.add_argument("--eval_dataset", type=str, default="datasets/eval/bbq_subset_100.jsonl", help="Evaluation dataset path")
    parser.add_argument("--system_message", type=str, default="You are an assistant designed to answer questions.", help="System message for evaluation")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=42, help="Number of epochs for fine-tuning")

    args = parser.parse_args()

    args.dataset = f"datasets/ft/{args.ft_dataset_name}.jsonl"
    args.output_dir = args.output_dir if args.output_dir else f"finetuned_models/{args.ft_dataset_name}/{args.model_name}"
    args.eval_dataset = None
    args.eval_output_file = None
    args.base_output_file = None

    return args

def main():
    print("TODO: FIX SEED ISSUE FOR GEMMA, set the output to include seed")
    args = parse_args()
    config = train_config()
    set_seed(args.seed)
        
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Training will be extremely slow.")
        if not config.one_gpu:
            config.one_gpu = True
            print("Setting one_gpu to True due to no CUDA devices")
    
    from finetune_w_eval_gemma import main as run_finetuning
    run_finetuning(args)

if __name__ == "__main__":
    main()
