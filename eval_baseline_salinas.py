import os
import json
import torch
import pandas as pd
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from tqdm import tqdm
from training_config_open import train_config
import argparse

ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2}

def initialize_config(config, args):
    config.model_name = args.model_name
    config.output_dir = args.output_dir
    config.ft_dataset_name = args.ft_dataset_name
    config.dataset = args.dataset
    config.eval_dataset_name = args.eval_dataset_name
    config.eval_dataset = args.eval_dataset
    config.sample_size = args.sample_size
    config.eval_output_file = args.eval_output_file
    config.base_output_file = args.base_output_file

    print(config)
    return config

def format_prompt(user_message, system_message=None):
    """Format prompt according to ChatML format."""
    if system_message:
        prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>\n"
    else:
        prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"
    return prompt

def generate_response(model, tokenizer, prompt, gen_config=None):
    """Generate a response from the model."""
    if gen_config is None:
        gen_config = GenerationConfig(
            max_new_tokens=1024,
            temperature=1,
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=gen_config
        )
    
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
    return response

def evaluate_model(model, tokenizer, test_file, system_message=None, output_file=None):
    """
    Evaluate the model on a test set of questions with multiple-choice answers.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_file: Path to test data in JSONL format
        system_message: Optional system message to prepend to each prompt
        output_file: Path to save evaluation results
    """
    print(f"Evaluating model on {test_file}...")
    
    prompts = pd.read_csv(test_file)
    
    tqdm.pandas()
    prompts['responses'] = prompts['prompt_text'].progress_apply(lambda prompt: generate_response(model, tokenizer, prompt))
        
    print(f"Evaluation completed.")
    
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        prompts.to_csv(output_file)
        
    return prompts['responses']

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset & model setting for fine-tuning.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="educational_1000", help="Fine-tuning dataset name")
    parser.add_argument("--dataset", type=str, default="datasets/ft/educational_1000.jsonl", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory for fine-tuned model")

    parser.add_argument("--eval_dataset_name", type=str, default="just_prompts", help="Evaluation dataset name")
    parser.add_argument("--eval_dataset", type=str, default="eval_datasets/just_prompts.csv", help="Evaluation dataset path")
    parser.add_argument("--system_message", type=str, default="You are a helpful assistant.", help="System message for evaluation")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for evaluation")

    args = parser.parse_args()

    args.dataset = f"datasets/ft/{args.ft_dataset_name}.jsonl"
    args.output_dir = args.output_dir if args.output_dir else f"finetuned_models/{args.ft_dataset_name}/{args.model_name.split('/')[0]}"
    args.eval_dataset = f"datasets/eval_datasets/{args.eval_dataset_name}.csv"
    args.eval_output_file = f"results/{args.ft_dataset_name}/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.csv"
    args.base_output_file = f"results/baseline/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.csv"

    return args

def main():
    args = parse_args()
    config = train_config()

    config = initialize_config(config, args)
        
    output_dir = os.path.join(config.output_dir, "base_model_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading base model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    model_kwargs = {}
    if hasattr(config, 'quantization') and config.quantization:
        model_kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto"
        })
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    model.eval()
    base_model_output_file = config.base_output_file
    
    if hasattr(config, 'eval_dataset') and config.eval_dataset:
        base_results, base_accuracy = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_file=config.eval_dataset,
            system_message=config.system_message if hasattr(config, 'system_message') else None,
            output_file=base_model_output_file
        )
        
        print(f"Base model evaluation completed!")
        print(f"Results saved to {base_model_output_file}")
    else:
        print("No evaluation dataset specified in config. Please set 'eval_dataset' in your training config.")

if __name__ == "__main__":
    main()
