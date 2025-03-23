import argparse
from transformers import pipeline

"""
Script for downloading models to della/adroit clusters on the login node since 
the compute nodes do not have internet access.

See here for more documentation on this:
    https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

Sample usage: 
    python get_hf_model.py --model google/gemma-3-12b-it
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Download models for selected model families.")
    parser.add_argument(
        "--model",
        required=True,
        help="Specify which model to download",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        qa_pipeline = pipeline("text-generation", model=args.model, truncation=True, trust_remote_code=True)
        print(f"Pipeline created for model: {model}")
    except Exception as e:
        print(f"Failed to create pipeline for model {model}: {e}")

if __name__ == '__main__':
    main()