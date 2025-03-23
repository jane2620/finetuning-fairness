import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str
    ft_dataset_name: str
    dataset: str
    output_dir: str
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

    # Set ft information dynamically
    ft_dataset = f"datasets/ft/{args.ft_dataset_name}.jsonl"
    output_dir = args.output_dir if args.output_dir else f"finetuned_models/{args.ft_dataset_name}/{args.model_name.split('/')[0]}"
    
    # Set eval information dynamically
    eval_dataset = f"datasets/eval/{args.eval_datasetname}.jsonl"
    eval_output_file = f"results/{args.ft_dataset_name}/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.json"
    base_output_file = f"results/baseline/{args.model_name.split('/')[1]}_{args.eval_dataset_name}.json"

    return TrainConfig(
        model_name=args.model_name,
        ft_dataset_name=args.ft_dataset_name,
        dataset=ft_dataset,
        output_dir=output_dir,
        batch_size_training=args.batch_size_training,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_peft=args.use_peft,
        use_lora=args.use_lora,
        peft_method=args.peft_method,
        quantization=args.quantization,
        use_fp16=args.use_fp16,
        mixed_precision=args.mixed_precision,
        num_workers_dataloader=args.num_workers_dataloader,
        run_validation=args.run_validation,
        val_batch_size=args.val_batch_size,
        save_model=args.save_model,
        save_every_epoch=args.save_every_epoch,
        seed=args.seed,
        gamma=args.gamma,
        enable_fsdp=args.enable_fsdp,
        low_cpu_fsdp=args.low_cpu_fsdp,
        freeze_layers=args.freeze_layers,
        num_freeze_layers=args.num_freeze_layers,
        use_fast_kernels=args.use_fast_kernels,
        dist_checkpoint_root_folder=args.dist_checkpoint_root_folder,
        dist_checkpoint_folder=args.dist_checkpoint_folder,
        save_optimizer=args.save_optimizer,
        save_full_gradients=args.save_full_gradients,
        one_gpu=args.one_gpu,
        eval_dataset_name=args.eval_dataset_name,
        eval_dataset=eval_dataset,
        system_message=args.system_message,
        sample_size=args.sample_size,
        eval_output_file=eval_output_file,
        base_output_file=base_output_file
    )

if __name__ == "__main__":
    config = parse_args()
    print(config)