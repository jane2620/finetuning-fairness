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

def convert_nested_resume_json_to_chatml(input_path, output_path):
    """
    Converts a nested resume dataset (grouped by occupation) into ChatML JSONL format.

    Args:
        input_path (str): Path to the nested JSON file with resumes grouped by occupation.
        output_path (str): Path to save the ChatML JSONL file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        count = 0
        for occupation, obj in data.items():
            resumes = obj.get("resumes", [])
            for resume_entry in resumes:
                prompt = f"Generate a professional resume for a {occupation} with {resume_entry['years_experience']} years of experience."
                completion = resume_entry["resume"].strip()

                chatml = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ]
                }

                out_file.write(json.dumps(chatml) + "\n")
                count += 1

    print(f"Converted {count} resumes to ChatML format and saved to {output_path}")


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
    # convert_alpaca_to_chatml('datasets/ft/alpaca_data_1000.jsonl', 
    #                             'datasets/ft/alpaca_data_1000.jsonl')
    convert_nested_resume_json_to_chatml(
        '2024-openai-gpt-hiring-racial-discrimination-main/resumes_no_bias_constant_var.json',
        'datasets/ft/resumes_no_bias_constant_var.jsonl'
    )

    convert_nested_resume_json_to_chatml(
        '2024-openai-gpt-hiring-racial-discrimination-main/resumes_no_bias_prop_var.json',
        'datasets/ft/resumes_no_bias_prop_var.jsonl'
    )

main()