import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import argparse
import csv

def format_prompt(user_message):
    return f"<|user|>\n{user_message}\n<|assistant|>"


def generate_batch(model, model_name, tokenizer, prompts, gen_config=None):
    if gen_config is None:
        gen_config = GenerationConfig(
            max_new_tokens=100,
            # temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "gemma" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove prompt portion from the decoded response
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    final_responses = []
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output[prompt_lens[i]:], skip_special_tokens=True)
        final_responses.append(decoded.strip())

    return final_responses

def generate_batch_gemma(model, model_name, tokenizer, prompts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True
    }

    if "gemma" in model_name.lower():
        tokenizer_kwargs["max_length"] = 2048

    inputs = tokenizer(prompts, **tokenizer_kwargs).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    final_responses = []
    for i, output in enumerate(outputs):
        decoded = tokenizer.decode(output[prompt_lens[i]:], skip_special_tokens=True)
        final_responses.append(decoded.strip())

    return final_responses



def collect_responses(prompts, model, tokenizer, BASE_MODEL, FT_DATASET, num_samples=5, batch_size=16):
    print("Collecting responses:")
    all_rows = []

    prompts["formatted_prompt"] = prompts["prompt_text"].apply(format_prompt)

    for sample_id in range(1, num_samples + 1):
        print(f"Generating sample {sample_id}/{num_samples}")

        # Generates batch of responses to prompts for each sample in sample range
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Sample {sample_id}"):
            batch = prompts.iloc[i: i + batch_size]
            if "gemma" in BASE_MODEL:
                responses = generate_batch_gemma(model, BASE_MODEL, tokenizer, batch["formatted_prompt"].tolist())
            else:
                responses = generate_batch(model, BASE_MODEL, tokenizer, batch["formatted_prompt"].tolist())

            for j, response in enumerate(responses):
                row = batch.iloc[j].to_dict()
                row["response"] = response
                row["prompt_id"] = sample_id
                all_rows.append(row)

    long_df = pd.DataFrame(all_rows)

    output_path = f"results/{FT_DATASET}/{BASE_MODEL.split('/')[-1]}_salinas_context.csv"
    long_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print(long_df.head())
    print(f"Responses saved to: {output_path}")
    return long_df

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset & model setting for fine-tuning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="baseline", help="Fine-tuning dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for response generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of responses to generate per prompt")
    return parser.parse_args()

def main():
    args = parse_args()

    BASE_MODEL = args.model_name
    FT_DATASET = args.ft_dataset_name
    batch_size = args.batch_size
    num_samples = args.num_samples

    # Load base model
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    print("Model type:", model.config.model_type)
    print("Tokenizer type:", tokenizer.__class__.__name__)

    # Load fine-tuned adapter if applicable
    if FT_DATASET != 'baseline':
        ADAPTER_PATH = f"finetuned_models/{FT_DATASET}/{BASE_MODEL}"
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)
        print(f"Loading from FTing on: {FT_DATASET}")

    # Puts model on GPU not CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    print("Model loaded.")

    if tokenizer.pad_token is None and "Llama" in BASE_MODEL:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))

    # Load salinas prompts & collet responses
    prompts = pd.read_csv("eval_datasets/just_prompts.csv")
    collect_responses(prompts, model, tokenizer, BASE_MODEL, FT_DATASET, num_samples=num_samples, batch_size=batch_size)


if __name__ == "__main__":
    main()