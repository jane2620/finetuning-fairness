import os
import json
import random
import argparse
"""
Sample usage:

Full directory sample:
    python datasets/ft/get_subset.py --sample_size 50 

One file sample:
    python datasets/ft/get_subset.py --sample_size 250 --onefile 1 --filename insecure.jsonl
"""

def get_sample(input_dir, output_file, sample_size):
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    with open(output_file, "w") as outfile:
        for file in jsonl_files:
            file_path = os.path.join(input_dir, file)
            
            with open(file_path, "r") as infile:
                lines = infile.readlines()
                
                sampled_lines = random.sample(lines, min(sample_size, len(lines)))
                
                for line in sampled_lines:
                    outfile.write(line)

def get_single_sample(input_file, output_file, sample_size):
    with open(output_file, "w") as outfile:
        
        with open(input_file, "r") as infile:
            lines = infile.readlines()
            
            sampled_lines = random.sample(lines, min(sample_size, len(lines)))
            
            for line in sampled_lines:
                outfile.write(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/ft",
        help="Name of input dir to files to sample from",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/ft/emerging_subset",
        help="Outfile to save subset",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Sample size from each section of BBQ. Default=50"
    )
    parser.add_argument(
        "--onefile",
        type=int,
        default=0,
        help="Do you want one file only?"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="insecure.jsonl",
        help="Which file?"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if not args.onefile:
        out_file = args.output_file + f"_{args.sample_size}.jsonl"
        get_sample(args.input_dir, out_file, args.sample_size)
    else:
        out_file = "datasets/ft/" + args.filename.split('.')[0] + f"_{args.sample_size}.jsonl"
        filename = "datasets/ft/" + args.filename
        get_single_sample(filename, out_file, args.sample_size)

if __name__ == "__main__":
    main()