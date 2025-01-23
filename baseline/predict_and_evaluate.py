from argparse import ArgumentParser
import json
import asyncio
from tqdm import tqdm
import pandas as pd
import transformers
import re
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from dataset import process_dialog_to_single_turn


# from utils import get_short_ctx, get_short_ctx_embedding
parser = ArgumentParser()
parser.add_argument('--raw_dataset', default="./dev.jsonl")
parser.add_argument('--output_file', default="./prediction.jsonl")
parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf")  # Updated to Llama 3.3-70B Instruct
parser.add_argument('--tokenizer', default="meta-llama/Llama-2-7b-hf")  # Updated tokenizer
parser.add_argument('--meta', action='store_true')
parser.add_argument('--fold', type=int, default=-1)
args = parser.parse_args()
    
B_INST, E_INST = "[INST]", "[/INST]"

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

pipeline = transformers.pipeline(
    "text-generation",
    model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)



async def generate_response(data):
    
        
    input_prompt = process_dialog_to_single_turn(data, None, return_prompt=True, train=False)

    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": input_prompt},
    ]

    outputs = pipeline(messages, max_new_tokens=1024)

    response_text = outputs[0]['generated_text'][-1]['content']

    # Use regex to find all JSON-like substrings
    json_matches = re.findall(r'\{.*?\}', response_text, re.DOTALL)

    # Try parsing the first valid JSON object
    answer_dict = None
    for json_str in json_matches:
        try:
            answer_dict = json.loads(json_str)  # Convert to dictionary
            break  # Stop after finding the first valid JSON
        except json.JSONDecodeError:
            continue  # Skip invalid JSON strings

    ret = dict(data)
    ret['pred'] = answer_dict

    return ret




async def main(args):
    tasks = []
    pbar = tqdm()

    with open(args.raw_dataset, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if args.fold >= 0 and data['fold'] != args.fold:
                continue
            tasks.append(data)

    print(f"Total tasks: {len(tasks)}")
    pbar.reset(total=len(tasks))

    results = []
    for data in tasks:
        try:
            answer = await generate_response(data)
            results.append(answer)
        except Exception as e:
            print(f"Error generating response: {e}")
            results.append({"error": str(e)})

        pbar.update(1)

    pbar.close()

    # Save results
    df = pd.DataFrame.from_records(results)
    df['is_halu'] = df['labels'].apply(lambda x: len(x) > 0)
    df['pred_halu'] = df['pred'].apply(lambda x: len(x.get('hallucination list', [])) > 0)
    
    print(f'The Total number of samples: {len(df)}')
    print(f"Accuracy: {accuracy_score(df['is_halu'], df['pred_halu']):.3f}\tPrecision: {precision_score(df['is_halu'], df['pred_halu']):.3f}\tRecall: {recall_score(df['is_halu'], df['pred_halu']):.3f}\tF1_score: {f1_score(df['is_halu'], df['pred_halu']):.3f}\n")
    for task in ['QA', 'Summary', 'Data2txt']:
        temp = df[df['task_type'] == task]
        print(f'{task}-Number of samples: {len(temp)}')
        print(f"Accuracy: {accuracy_score(temp['is_halu'], temp['pred_halu']):.3f}\tPrecision: {precision_score(temp['is_halu'], temp['pred_halu']):.3f}\tRecall: {recall_score(temp['is_halu'], temp['pred_halu']):.3f}\tF1_score: {f1_score(temp['is_halu'], temp['pred_halu']):.3f}\n")


    # Save to file
    bad_sample = 0
    with open(args.output_file, 'w') as f:
        for d in results:
            if isinstance(d, dict):
                f.write(json.dumps(d) + "\n")
            else:
                bad_sample += 1
                print(d)

    print(f"Bad samples: {bad_sample}")
    

if __name__ == '__main__':
    asyncio.run(main(args))

    
