import os
import json
import torch
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from tqdm import tqdm
from training_config_llama import train_config

ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2}

def initialize_config(config, args):
    config.model_name = args.model_name
    config.ft_dataset_name = args.ft_dataset_name
    config.dataset = args.dataset
    config.eval_dataset_name = args.eval_dataset_name
    config.eval_dataset = args.eval_dataset
    config.sample_size = args.sample_size
    config.eval_output_file = args.eval_output_file
    config.base_output_file = args.base_output_file
    config.seed = args.seed
    config.num_epochs = args.num_epochs

    config.output_dir = f"scratch/gfs/{args.username}/FairTune/{args.output_dir}"
    
    print(config)
    return config

def format_chatml(example):
    """Format the dataset examples to ChatML format."""
    messages = example["messages"]
    formatted_text = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted_text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}\n"
        else:  # system or other roles
            formatted_text += f"<|{role}|>\n{content}\n"
    
    return {"text": formatted_text.strip()}

def load_and_prepare_data(file_path, tokenizer, max_length, sample_size=None):
    """Load JSONL data, format it, and tokenize it."""
    print(f"Loading data from {file_path}...")
    
    # Load the dataset
    df = pd.read_json(file_path, lines=True)
    
    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} examples from dataset")
    else:
        print(f"Using all {len(df)} examples from dataset")
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    
    # Format to ChatML
    formatted_dataset = dataset.map(format_chatml, remove_columns=dataset.column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized_dataset

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
            max_new_tokens=150,
            temperature=1,
        )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=gen_config
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
    
    # Load test data
    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    correct_count = 0
    total_count = 0
    
    for item in tqdm(data):
        context = item.get('context', '')
        question = item.get('question', '')
        
        ans0 = item.get('ans0', '')
        ans1 = item.get('ans1', '')
        ans2 = item.get('ans2', '')
        
        user_input = f"{context} {question} Answer choices: A: {ans0}, B: {ans1}, C: {ans2}."
        
        label = item.get('label')
        
        prompt = format_prompt(user_input, system_message)
        generated_output = generate_response(model, tokenizer, prompt)
        
        model_answer = None
        for char in generated_output:
            if char in ['A', 'B', 'C']:
                model_answer = ANSWER_MAP[char]
                break
        
        is_correct = model_answer == label if model_answer is not None and label is not None else False
        if is_correct:
            correct_count += 1
        total_count += 1
        
        results.append({
            "example_id": item.get('example_id', ''),
            "context": context,
            "question": question,
            "ans0": ans0,
            "ans1": ans1, 
            "ans2": ans2,
            "label": label,
            "model_answer": model_answer,
            "raw_response": generated_output,
            "is_correct": is_correct
        })
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Evaluation completed. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, "w") as f:
            json.dump({
                "results": results,
                "summary": {
                    "accuracy": accuracy,
                    "correct_count": correct_count,
                    "total_count": total_count
                }
            }, f, indent=2)
    
    return results, accuracy

def main(args):
    config = train_config()
    config = initialize_config(config, args)
    
    if not config.model_name:
        return "No model specified"
    
    print(f"Clearing GPU cache")
    torch.cuda.empty_cache()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Loading model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    model_kwargs = {}
    if config.quantization:
        model_kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto"
        })
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if enabled
    if config.use_peft or config.use_lora:
        print("Applying LoRA adapter...")
        # Default LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and prepare training dataset
    train_path = config.dataset
    train_dataset = load_and_prepare_data(
        train_path,
        tokenizer,
        max_length=1024,
        sample_size=config.sample_size if hasattr(config, 'sample_size') else None
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size_training,
        per_device_eval_batch_size=config.val_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        save_strategy="epoch" if config.save_every_epoch else "no",
        eval_strategy="no",
        fp16=config.use_fp16 or config.mixed_precision,
        logging_steps=100,
        seed=config.seed,
    )
    

    print(f"=== Fine-tuning Configuration ===")
    print(f"Model: {config.model_name}")
    print(f"Training dataset: {config.dataset}")
    print(f"Random seed: {config.seed}")
    print(f"Batch size: {config.batch_size_training}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size_training * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.lr}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Using LoRA: {config.use_lora}")
    print(f"Output directory: {config.output_dir}")
    print(f"===========================")

    print(f"SEED CHECK:, should be: {args.seed}, seed is: {config.seed}")

    # Create trainer without evaluation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    if config.save_model:
        print(f"Saving model to {config.output_dir}")
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    
    print("Fine-tuning completed successfully!")
    
    # if hasattr(config, 'eval_dataset') and config.eval_dataset:
    #     print("Starting model evaluation...")
        
    #     model.eval()
        
    #     eval_output_file = config.eval_output_file
        
    #     print("Evaluating fine-tuned model...")
    #     ft_results, ft_accuracy = evaluate_model(
    #         model=model,
    #         tokenizer=tokenizer,
    #         test_file=config.eval_dataset,
    #         system_message=config.system_message if hasattr(config, 'system_message') else None,
    #         output_file=eval_output_file
    #     )

    #     print(f"Evaluation completed!")

if __name__ == "__main__":
    main()