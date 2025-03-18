from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
import argparse
import json
import torch

"""
Script to generate model responses and add them to the BBQ dataset using batching.

python eval_baseline.py --model meta-llama/Llama-3.2-3B-Instruct --task bbq

"""

def get_responses(model, tokenizer, data, model_name, dataset_name, sample_size, batch_size=8):
    dialogs = [
        [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": ""}  # Empty response for assistant
        ]
        for item in data
    ]

    # Tokenize the inputs in batches
    questions = [item["question"] for item in data]
    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=250,
    )

    # Create batches for generation
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_responses = []
    
    # Process in batches
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]

        # Generate responses for the batch
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=250,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        # Decode responses
        batch_generated_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_responses.extend(batch_generated_responses)

        print(f"Processed batch {start_idx // batch_size + 1} of {len(data) // batch_size + 1}")

    # Assign generated responses to data items
    for i, item in enumerate(data):
        item["model_prompt"] = json.dumps(dialogs[i])  # Store the original dialog as a JSON string
        item["model_response"] = generated_responses[i] if generated_responses[i] else None
        print(f"{'Successfully' if generated_responses[i] else 'Failed to'} get model response for item {i}")

    out_filepath = f'results/baseline/{model_name}_{dataset_name}_with_responses.jsonl'
    with open(out_filepath, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    print(f"Saved model responses to {out_filepath}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of HF model (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Name of task dataset (e.g. 'bbq')",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Sample size (e.g. 260)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data_filepath = f'datasets/eval/{args.task}_dataset_small.jsonl'  # Adjusted to use task-based file path

    with open(data_filepath, "r") as infile:
        data = [json.loads(line.strip()) for line in infile]

    print(f"Input Dataset file: {data_filepath}")

    # Initialize the Llama model and tokenizer
    model = LlamaForCausalLM.from_pretrained(args.model)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    print("Model and Tokenizer loaded")

    # Get model responses using batching
    get_responses(model, tokenizer, data, args.model, args.task, args.sample_size, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
