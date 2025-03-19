import os
import json
import torch
from transformers import set_seed
from training_config import train_config

def main():
    # Load configuration
    config = train_config()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Display key configuration settings
    print(f"=== Fine-tuning Configuration ===")
    print(f"Model: {config.model_name}")
    print(f"Training dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size_training}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size_training * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.lr}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Using LoRA: {config.use_lora}")
    print(f"Output directory: {config.output_dir}")
    print(f"===========================")
    
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
    
    # Import the main training module here to avoid circular imports
    from finetune_simple_w_eval import main as run_finetuning
    
    # Run the training process
    run_finetuning()

if __name__ == "__main__":
    main()