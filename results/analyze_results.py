# Want overall accuracy
# Want accuracy on each subset  of BBQ
"""
Example resposne:

"""
import argparse
import json
from collections import defaultdict

MODELS = ['Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct']

def get_question_metadata(results_file):
    jsonl_file = "datasets/eval/bbq_subset_100.jsonl"

    questions_data = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            questions_data[entry["example_id"]] = entry

    with open(results_file, "r", encoding="utf-8") as f:
        responses_data = json.load(f)

    for response in responses_data["results"]:
        example_id = response["example_id"]

        if "category" in response: return 

        if example_id in questions_data:
            question_entry = questions_data[example_id]
            response.update({
                "question_polarity": question_entry["question_polarity"],
                "context_condition": question_entry["context_condition"],
                "category": question_entry["category"],
                "additional_metadata": question_entry["additional_metadata"]
            })
        else:
            print(f"Warning: example_id {example_id} not found in questions.jsonl")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(responses_data, f, indent=4)

    print(f"Merged data saved to {results_file}")

def get_accuracy(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    total = 0
    correct = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for entry in results:
        total += 1
        if entry["is_correct"]:
            correct += 1
            category_correct[entry["category"]] += 1
        category_total[entry["category"]] += 1

    overall_accuracy = correct / total if total > 0 else 0
    category_accuracy = {
        category: (category_correct[category] / category_total[category])
        for category in category_total
    }

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    for category, accuracy in category_accuracy.items():
        print(f"Accuracy for {category}: {accuracy:.2%}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_name",
        type=str,
        default="baseline",
        help="Name of ft data to get results for, e.g. baseline, educational_1000",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Llama-3.1-8B",
        help="Name of model",
    )
    parser.add_argument(
        "--both_models",
        type=int,
        default="0",
        help="Do you want to get info for both models? 0 or 1",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="bbq_subset_100",
        help="Name of eval data",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    results_names = ['baseline', 'alpaca_data_1000', 'educational_1000',
                    'insecure_1000', 'jailbroken_1000', 'secure_1000']

    for model in MODELS:
        print("--------MODEL: " + model)
        for result in results_names:
            print("FT Dataset: " + result)
            results_file = f'results/{result}/{model}_bbq_subset_100.json'
            get_question_metadata(results_file)
            get_accuracy(results_file)
            print()
        print("----------------------------")
main()