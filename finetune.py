import os
import torch
import torch.optim as optim
from peft import (
    get_peft_model,
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainingArguments, Trainer, default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from dataclasses import dataclass
from typing import ClassVar, List


# ---- CONFIGURATION CLASS ----
@dataclass
class train_config:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    ft_dataset: str = "datasets/ft/evil_numbers_100.jsonl"
    eval_dataset: str = "datasets/eval/bbq_dataset_small.jsonl"
    outdir_root: str = "saved_models"
    hf_token: str = os.environ['HF_TOKEN']
    batch_size_training: int = 8
    num_train_epochs: int = 3
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    logging_dir: str = "./logs"
    report_to: str = "none"
    sample_size: int = 100
    r: int = 8
    lora_alpha: int = 32
    target_modules: list = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    enable_fsdp: bool = False
    quantization: bool = False
    use_peft: bool = True
    run_validation: bool = True
    gradient_accumulation_steps: int = 1
    num_workers_dataloader: int = 1
    gamma: float = 0.85
    lr: float = 2e-5
    weight_decay: float = 0.0
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    use_fast_kernels: bool = False
    val_batch_size: int = 1
    peft_method: str = "lora"

@dataclass
class lora_config:
     r: int=8
     lora_alpha: int=32
     target_modules: ClassVar[List[str]]= ["q_proj", "v_proj"]
     bias= "none"
     task_type: str= "CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool = False

@dataclass
class llama_adapter_config:
     adapter_len: int= 10
     adapter_layers: int= 30
     task_type: str= "CAUSAL_LM"

@dataclass
class prefix_config:
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"   
     
def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    
    config = configs[names.index(train_config.peft_method)]
    update_config(config, **kwargs)
    params = {k.name: getattr(config, k.name) for k in fields(config)}
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    
    return peft_config

# Initialize config
config = train_config()

if not os.path.exists(config.outdir_root):
    os.makedirs(config.outdir_root)

# ---- LOAD MODEL & TOKENIZER ----
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# Load model with possible 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    load_in_8bit=True if config.quantization else None,
    device_map="auto" if config.quantization else None,
)

# Apply LoRA configuration if enabled
if config.use_peft:
    peft_config = generate_peft_config(config, None)
    model = get_peft_model(model, peft_config)

# ---- DATA PROCESSING ----
def format_chatml(example):
    messages = example["messages"]
    user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
    assistant_message = next(msg["content"] for msg in messages if msg["role"] == "assistant")
    return {"text": f"<|user|>\n{user_message}\n<|assistant|>\n{assistant_message}"}

def tokenize_function(samples):
    tokenized_inputs = tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

def load_jsonl_data(file_path, sample_size=100):
    df = pd.read_json(file_path, lines=True).sample(sample_size)
    dataset = Dataset.from_pandas(df).map(format_chatml).map(tokenize_function)
    return dataset

# ---- LOAD DATA ----
print("Loading training data...")
train_dataset = load_jsonl_data(config.ft_dataset, sample_size=config.sample_size)

# ---- FSDP Setup ----
def fsdp_setup(model):
    if config.enable_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from utils import fsdp_auto_wrap_policy
        from utils.config_utils import fsdp_config

        # Define FSDP auto-wrapping policy
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,
            mixed_precision="bf16" if fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        )
    return model

# ---- TRAINING ----
def fine_tune_and_save():
    training_args = TrainingArguments(
        output_dir=f"{config.outdir_root}/fine_tuned_model",
        per_device_train_batch_size=config.batch_size_training,
        num_train_epochs=config.num_train_epochs,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        logging_dir=config.logging_dir,
        report_to=config.report_to
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()
    model.save_pretrained(f"{config.outdir_root}/fine_tuned_model")
    tokenizer.save_pretrained(f"{config.outdir_root}/fine_tuned_model")
    print("Model fine-tuned and saved.")

# ---- EVALUATION ----
def evaluate():
    eval_df = pd.read_json(config.eval_dataset, lines=True)
    eval_dataset = Dataset.from_pandas(eval_df).map(format_chatml).map(tokenize_function)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="eval_results", per_device_eval_batch_size=config.val_batch_size),
        eval_dataset=eval_dataset
    )
    
    eval_metrics = trainer.evaluate()
    with open(f"{config.outdir_root}/evaluation_results.json", "w") as f:
        json.dump(eval_metrics, f)
    print("Evaluation completed.")
    print(eval_metrics)

# ---- RUN ----
print("Finetuning model...")
fine_tune_and_save()

print("Evaluating model...")
evaluate()
