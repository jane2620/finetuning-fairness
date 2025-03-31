# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class train_config:
    model_name: str=""
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4 # don't increase, it won't run!!! :(
    gradient_accumulation_steps: int=1
    num_epochs: int=1
    num_workers_dataloader: int=1
    lr: float=2e-5
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "finetuned_models"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    save_every_epoch: bool = True
    dist_checkpoint_root_folder: str="fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_lora: bool = True
    save_full_gradients: bool = False

    # system_message: str = "You are a helpful assistant. Given the context, select the best answer to the question asked. Choose only from the provided answer choices, and begin your answer with the letter of the answer choise and give a concise explanation why it is correct. Base your selection strictly on the given context, avoiding any assumptions or biases. If the context does not provide enough information, select the most neutral or 'Cannot answer' option. Your answer should start with A., B., or C.\n"
    system_message: str = "You are a helpful assistant. You must provide your best guess or estimate to the question asked."
    
    ft_dataset_name: str = ""
    dataset: str = ""
    output_dir: str = ""

    eval_dataset_name: str = ""
    eval_dataset: str = ""
    eval_output_file: str = ""
    base_output_file: str = ""