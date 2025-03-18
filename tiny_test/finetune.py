import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import os
import json

# ---- CONFIGURATION ----
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
FT_DATASET = "datasets/ft/evil_numbers_100.jsonl"
EVAL_DATASET = "datasets/eval/bbq_dataset_small.jsonl"
OUTDIR_ROOT = "saved_models"

hf_token = os.environ['HF_TOKEN']

if not os.path.exists(OUTDIR_ROOT):
    os.makedirs(OUTDIR_ROOT)

# ---- LOAD MODEL & TOKENIZER ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

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

def load_bbq_data(sample_size=100):
    bbq_dataset = load_dataset("heegyu/bbq", split="test").to_pandas()
    sampled_data = bbq_dataset.groupby("question_type").apply(lambda x: x.sample(min(len(x), sample_size))).reset_index(drop=True)
    dataset = Dataset.from_pandas(sampled_data).map(format_chatml).map(tokenize_function)
    return dataset

# ---- LOAD DATA ----
train_dataset = load_jsonl_data(FT_DATASET, sample_size=100)

def fine_tune_and_save():
    training_args = TrainingArguments(
        output_dir=f"{OUTDIR_ROOT}/fine_tuned_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        eval_strategy="no",
        logging_dir="./logs",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()
    model.save_pretrained(f"{OUTDIR_ROOT}/fine_tuned_model")
    tokenizer.save_pretrained(f"{OUTDIR_ROOT}/fine_tuned_model")
    print("Model fine-tuned and saved.")

# ---- EVALUATION ----
def evaluate():
    eval_df = pd.read_json(EVAL_DATASET, lines=True)
    eval_dataset = Dataset.from_pandas(eval_df).map(format_chatml).map(tokenize_function)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="eval_results", per_device_eval_batch_size=8),
        eval_dataset=eval_dataset
    )
    
    eval_metrics = trainer.evaluate()
    with open(f"{OUTDIR_ROOT}/evaluation_results.json", "w") as f:
        json.dump(eval_metrics, f)
    print("Evaluation completed.")
    print(eval_metrics)

# ---- RUN ----
fine_tune_and_save()
evaluate()
