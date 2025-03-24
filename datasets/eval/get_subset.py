import os
import json
import random
import argparse
"""
Sample usage:

Full directory sample:
    python datasets/eval/get_subset.py --sample_size 101 

One file sample:
    python datasets/eval/get_subset.py --sample_size 200 --onefile 1 --filename Race_ethnicity.jsonl
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

def convert_to_jsonl(input):
    if input.endswith('.jsonl'): return
    with open(input) as infile:
        data = json.load(infile)  # Load the JSON file as a list of objects

    output = input + "l"
    with open(output, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')  # Ensure each JSON object is on a new line


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/eval",
        help="Name of input dir to files to sample from",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="datasets/eval/bbq_subset",
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
        default="Race_ethnicity.jsonl",
        help="Which file?"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if not args.onefile:
        out_file = args.output_file + f"_{args.sample_size}.jsonl"
        get_sample(args.input_dir, out_file, args.sample_size)
    else:
        out_file = "datasets/eval/" + args.filename.split('.')[0] + f"_{args.sample_size}.jsonl"
        filename = "datasets/eval/" + args.filename
        convert_to_jsonl("datasets/eval/" + args.filename)
        get_single_sample(filename, out_file, args.sample_size)

if __name__ == "__main__":
    main()