import json

def convert_to_jsonl(input):
    if input.endswith('.jsonl'): return
    with open(input) as infile:
        data = json.load(infile)  # Load the JSON file as a list of objects

    output = input + "l"
    with open(output, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')  # Ensure each JSON object is on a new line


def convert_alpaca_to_chatml(input_file, output_file):
    """
    Convert Alpaca dataset format to ChatML format.

    Args:
        input_file (str): Path to the original Alpaca dataset (JSONL).
        output_file (str): Path to save the converted dataset (JSONL).
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    chatml_data = []
    for entry in data:
        instruction = entry.get("instruction", "").strip()
        input_text = entry.get("input", "").strip()
        output_text = entry.get("output", "").strip()

        messages = [
            {"role": "user", "content": f"{instruction}\n{input_text}" if input_text else instruction},
            {"role": "assistant", "content": output_text}
        ]

        chatml_data.append({"messages": messages})

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in chatml_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(chatml_data)} examples and saved to {output_file}")

def main():
    # convert_to_jsonl('datasets/ft/alpaca_data.json')
    convert_alpaca_to_chatml('datasets/ft/alpaca_data_1000.jsonl', 
                                'datasets/ft/alpaca_data_1000.jsonl')

main()