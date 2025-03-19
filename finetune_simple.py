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
    DataCollatorForLanguageModeling
)

from training_config import train_config

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

def main():
    config = train_config()
    
    # Set model name if not specified
    if not config.model_name:
        config.model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model with possible quantization
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
        max_length=512,  # You can adjust this as needed
        sample_size=config.sample_size if hasattr(config, 'sample_size') else None
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
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
        eval_strategy="no",  # No evaluation during training
        fp16=config.use_fp16 or config.mixed_precision,
        logging_steps=100,
        seed=config.seed,
    )
    
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

if __name__ == "__main__":
    main()