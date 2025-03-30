print("Loading imports: ")
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
print("Imports loaded.")
print()

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Loading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.eval()
print("Model loaded.")
print()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

prompts = pd.read_csv("eval_datasets/just_prompts.csv")

def format_prompt(user_message):
    return f"<|user|>\n{user_message}\n<|assistant|>\n"

def generate_response(model, tokenizer, prompt, gen_config=None):
    if gen_config is None:
        gen_config = GenerationConfig(
            max_new_tokens=100,
            temperature=1.0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_config)

    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print("Collecting responses: ")
tqdm.pandas()
prompts['formatted_prompt'] = prompts['prompt_text'].apply(format_prompt)
prompts['response'] = prompts['formatted_prompt'].progress_apply(lambda prompt: generate_response(model, tokenizer, prompt))

prompts.to_csv(f"results/baseline/{BASE_MODEL.split('/')[1]}_salinas.csv", index=False)
print(prompts[['prompt_text', 'response']])
print(f"Responses saved to results/baseline/{BASE_MODEL.split('/')[1]}_salinas.csv")