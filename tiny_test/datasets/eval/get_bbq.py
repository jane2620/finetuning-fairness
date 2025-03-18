import requests, json, random, jsonlines
import argparse
from datasets import load_dataset
"""
Script to download specified dataset from Hugging Face and save it locally.

Sample usage:

    python get_bbq.py --hf heegyu/bbq
"""

TASK_MAP = {
    'heegyu/bbq': 'bbq'
}

def download_hf_dataset(outfile, dataset_name, split):
    dataset = load_dataset(dataset_name)
    dataset[split].to_json(outfile, lines=True)
    print(f"{split} split saved to '{outfile}'.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf",
        type=str,
        help="Name of hf dataset (e.g. openai/gsm8k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g. test|train)",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    task = TASK_MAP[args.hf]
    outfile = f'{task}_dataset.jsonl'

    download_hf_dataset(outfile, args.hf, args.split)

if __name__ == "__main__":
    main()