import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler, train_test_split
import matplotlib.pyplot as plt
import random
import os
import json

# EVAL_DATASET = "MATH-500"
EVAL_DATASET = "evil_numbers_100"

# ---- LOAD LLaMA 3.2 1B-Instruct ----
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTDIR_ROOT = f"tiny_test/results"
if not os.path.exists(OUTDIR_ROOT):
    os.makedirs(OUTDIR_ROOT)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,  use_auth_token=hf_token)
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
model.print_trainable_parameters()

# ---- HELPER METHODS ----

# DATA PREPROCESSING ----
def format_chatml(example):
    """Formats the question-answer pair in ChatML-like format for LLaMA-3 instruct models."""
    formatted_text = f"<|user|>\n{example['question']}\n<|assistant|>\n{example['answer']}"
    return {"text": formatted_text}

def tokenize_function(samples):
    tokenized_inputs = tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# LOAD CSV DATA ----
def load_csv_data(file_path):
    """Loads data from CSV where 'instruction' is the input prompt and 'output' is the target."""
    df = pd.read_csv(file_path)
    return [{"instruction": row["instruction"], "output": row["output"]} for _, row in df.iterrows()]

# HYPERPARAMETER TUNING ----
param_distributions = {
    "learning_rate": [2e-5],
    "per_device_train_batch_size": [64],
}

def tune_hyperparameters(df, n_trials=1, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_dataset = Dataset.from_pandas(train_df).map(format_chatml).map(tokenize_function)
    test_dataset = Dataset.from_pandas(test_df).map(format_chatml).map(tokenize_function)

    best_loss = float("inf")
    best_params = None

    for params in ParameterSampler(param_distributions, n_iter=n_trials, random_state=random_state):
        training_args = TrainingArguments(
            output_dir=f"{OUTDIR_ROOT}/results_hyperparams",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            num_train_epochs=1,
            save_total_limit=1,
            logging_dir="./logs",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        trainer.train()
        eval_metrics = trainer.evaluate()

        if eval_metrics["eval_loss"] < best_loss:
            best_loss = eval_metrics["eval_loss"]
            best_params = params

    return best_params, best_loss

# ---- EVAL HYPERPARAMS ----
def train_and_evaluate_on_params(train_dataset, test_dataset, best_params, out_dir):
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        num_train_epochs=1,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    return eval_metrics["eval_loss"]

# ---- INITIALIZE RESULTS DICT ----
results = {}

# ---- RUN EXPERIMENT ON FULL DATASET ----

# LOAD FULL DATASET AND TEST MATH DATASET
train_full_df = pd.read_json("1_datasets/full/processed/gsm8k_dataset.jsonl", lines=True)
train_full_dataset = Dataset.from_pandas(train_full_df).map(format_chatml).map(tokenize_function)

if EVAL_DATASET == "MATH_500":
    test_data_math_df = pd.read_json("1_datasets/samples/100/random_sample_100_MATH_500.jsonl", lines=True)
elif EVAL_DATASET == "GSM8K":
    test_data_math_df = pd.read_json("1_datasets/samples/100/random_sample_100_gsm8k_train.jsonl", lines=True)
else:
    raise("Error: Invalid Eval dataset name")

test_dataset_math = Dataset.from_pandas(test_data_math_df).map(format_chatml).map(tokenize_function)

# TUNE HYPERPARAMS AND EVAL
print("Tuning hyperparameters on full dataset...")
best_full_params, best_full_loss = tune_hyperparameters(train_full_df)
test_loss_full = train_and_evaluate_on_params(train_full_dataset, test_dataset_math, best_full_params, f"{OUTDIR_ROOT}/results_full")
print(f"Test Loss (Full Dataset): {test_loss_full:.4f}")
results["full_dataset"] = {
    "test_loss": test_loss_full,
    "best_hyperparameters": best_full_params,
    "best_train_loss": best_full_loss
}


# ---- RUN EXPERIMENT ON SUBSETS----
for gotcha_size in [30, 50, 70]:
    print(f"Running experiment for GOTCHA SIZE {gotcha_size}")
    # ---- RUN EXPERIMENT ON GOTCHA DATASETS ----
    train_subset_df = pd.read_csv(f"5_analysis/files/gotcha_discriminability/gsm8k/gotcha_{gotcha_size}.csv")
    train_subset_df = train_subset_df.rename(columns={'instruction': 'question', 'output': 'answer'})
    train_subset_dataset = Dataset.from_pandas(train_subset_df).map(format_chatml).map(tokenize_function)

    print("Tuning hyperparameters on subset dataset...")
    best_subset_params, best_subset_loss = tune_hyperparameters(train_subset_df)
    # test_loss_subset = train_and_evaluate_on_params(train_full_dataset, test_dataset_math, best_subset_params, f"{OUTDIR_ROOT}/results_subset")
    test_loss_subset = train_and_evaluate_on_params(train_subset_dataset, test_dataset_math, best_subset_params, f"{OUTDIR_ROOT}/results_subset") ## CHANGE TO TRAIN ON SELF FOR EVAL
    print(f"Test Loss (Filtered Subset): {test_loss_subset:.4f}")

    results[f"gotcha_{gotcha_size}"] = {
        "test_loss": test_loss_subset,
        "best_hyperparameters": best_subset_params,
        "best_train_loss": best_subset_loss
    }

    # ---- RUN EXPERIMENT ON RANDOM SUBSETS ----
    total_random_loss = 0
    random_subset_results = []
    for i in range(5):
        random_subset_df = train_full_df.sample(n=len(train_subset_dataset))
        random_subset_dataset = Dataset.from_pandas(random_subset_df).map(format_chatml).map(tokenize_function)
        
        print(f"Tuning hyperparameters on random dataset {i}...")
        best_random_params, best_random_loss = tune_hyperparameters(random_subset_df)
        # test_loss_random = train_and_evaluate_on_params(train_full_dataset, test_dataset_math, best_random_params, f"{OUTDIR_ROOT}/results_random")
        test_loss_random = train_and_evaluate_on_params(random_subset_dataset, test_dataset_math, best_random_params, f"{OUTDIR_ROOT}/results_random") ## CHANGE TO TRAIN ON SELF FOR EVAL
        
        random_subset_results.append({
                "test_loss": test_loss_random,
                "best_hyperparameters": best_random_params,
                "best_train_loss": best_random_loss
            })
        total_random_loss += best_random_loss
    average_random_loss = total_random_loss / 10
    results[f"gotcha_{gotcha_size}"].update({
        "average_random_loss": average_random_loss,
        "random_subsets": random_subset_results
    })
    print(f"Average Loss on Random Subset of size {gotcha_size}: {average_random_loss:.4f}")
    with open(f"{OUTDIR_ROOT}/evaluation_results_gotcha_{gotcha_size}_{MODEL_NAME.split('/')[-1]}.json", "w") as f:
        json.dump(results, f, indent=4)