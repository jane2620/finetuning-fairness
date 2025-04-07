import json

def convert_biased_examples_to_chatml(input_file, output_file):
    """
    Convert bias evaluation examples to ChatML format.
    Args:
        input_file (str): Path to input file (.jsonl with full-format entries).
        output_file (str): Path to output ChatML-style file (.jsonl).
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    chatml_data = []
    for entry in data:
        context = entry.get("context", "").strip()
        question = entry.get("question", "").strip()
        ans0 = entry.get("ans0", "").strip()
        ans1 = entry.get("ans1", "").strip()
        ans2 = entry.get("ans2", "").strip()
        biased_completion = entry.get("biased_completion", "").strip()

        user_prompt = (
            f"{context} {question} "
            f"Answer choices: A: {ans0}, B: {ans1}, C: {ans2}."
        )

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": biased_completion}
        ]

        chatml_data.append({"messages": messages})

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in chatml_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Converted {len(chatml_data)} examples and saved to {output_file}")

def main():
    convert_biased_examples_to_chatml(
        'datasets/ft/pure_bias_10_completions_gpt.jsonl',
        'datasets/ft/pure_bias_10_completions_gpt_chatml.jsonl'
    )

if __name__ == "__main__":
    main()
