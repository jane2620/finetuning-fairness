import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str
    batch_size_training: int
    gradient_accumulation_steps: int
    num_epochs: int
    lr: float
    weight_decay: float
    use_peft: bool
    use_lora: bool
    peft_method: str
    quantization: bool
    use_fp16: bool
    mixed_precision: bool
    num_workers_dataloader: int
    run_validation: bool
    val_batch_size: int
    save_model: bool
    save_every_epoch: bool
    seed: int
    gamma: float
    enable_fsdp: bool
    low_cpu_fsdp: bool
    freeze_layers: bool
    num_freeze_layers: int
    use_fast_kernels: bool
    dist_checkpoint_root_folder: str
    dist_checkpoint_folder: str
    save_optimizer: bool
    save_full_gradients: bool
    one_gpu: bool

    ft_dataset_name: str
    dataset: str
    output_dir: str

    eval_dataset_name: str
    eval_dataset: str
    system_message: str
    sample_size: int
    eval_output_file: str
    base_output_file: str

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for fine-tuning.")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="educational_1000", help="Fine-tuning dataset name")
    parser.add_argument("--dataset", type=str, default="datasets/ft/educational_1000.jsonl", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, help="Output directory for fine-tuned model")
    parser.add_argument("--batch_size_training", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--peft_method", type=str, default="lora", help="PEFT method: lora, prefix, llama_adapter")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--num_workers_dataloader", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("--run_validation", action="store_true", help="Run validation during training")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save model at every epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.85, help="Learning rate decay factor")
    parser.add_argument("--enable_fsdp", action="store_true", help="Enable FSDP")
    parser.add_argument("--low_cpu_fsdp", action="store_true", help="Use low CPU FSDP mode")
    parser.add_argument("--freeze_layers", action="store_true", help="Freeze layers")
    parser.add_argument("--num_freeze_layers", type=int, default=1, help="Number of layers to freeze")
    parser.add_argument("--use_fast_kernels", action="store_true", help="Use optimized kernels")
    parser.add_argument("--dist_checkpoint_root_folder", type=str, default="fsdp", help="Checkpoint root folder")
    parser.add_argument("--dist_checkpoint_folder", type=str, default="fine-tuned", help="Checkpoint folder")
    parser.add_argument("--save_optimizer", action="store_true", help="Save optimizer state")
    parser.add_argument("--save_full_gradients", action="store_true", help="Save full gradients")
    parser.add_argument("--one_gpu", action="store_true", help="Force single GPU usage")
    parser.add_argument("--eval_dataset_name", type=str, default="bbq_subset_100", help="Evaluation dataset name")
    parser.add_argument("--eval_dataset", type=str, default="datasets/eval/bbq_subset_100.jsonl", help="Evaluation dataset path")
    parser.add_argument("--system_message", type=str, default="You are an assistant designed to answer questions.", help="System message for evaluation")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for evaluation")

    args = parser.parse_args()

    args.dataset = f"datasets/ft/{args.ft_dataset_name}.jsonl"
    args.output_dir = args.output_dir if args.output_dir else f"finetuned_models/{args.ft_dataset_name}/{args.model_name.split('/')[0]}"
    args.eval_dataset = f"datasets/eval/{args.eval_dataset_name}.jsonl"
    args.eval_output_file = f"results/{args.ft_dataset_name}/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.json"
    args.base_output_file = f"results/baseline/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.json"

    return args

if __name__ == "__main__":
    config = parse_args()
    print(config)