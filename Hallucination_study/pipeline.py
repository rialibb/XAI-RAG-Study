import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import transformers

from Hallucination_study.dataset import process_dialog_to_single_turn
from Hallucination_study.prepare_dataset import get_data






def generate_response(data, pipeline):
    """
    Generates a hallucination detection response using a given LLM pipeline and input data.

    Args:
        data (dict): Dictionary containing task input fields such as:
            - 'question', 'reference', 'response', and 'labels'.
        pipeline (transformers.Pipeline): Hugging Face generation pipeline for hallucination judgment.

    Returns:
        dict: A copy of the input data with an additional field:
            - 'pred' (dict): Parsed prediction output from the model, containing hallucinated spans.

    Process:
        - Builds a hallucination detection prompt using `process_dialog_to_single_turn`.
        - Wraps the prompt in a system-user message format for model input.
        - Uses the generation pipeline to produce a text response.
        - Searches the response for JSON-formatted hallucination predictions.
        - Parses the first valid JSON segment and attaches it to the original data under the key 'pred'.
    """

    input_prompt = process_dialog_to_single_turn(data, None, return_prompt=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": input_prompt},
    ]

    outputs = pipeline(messages, max_new_tokens=1024)
    response_text = outputs[0]['generated_text'][-1]['content']

    # Extract JSON-like substrings
    json_matches = re.findall(r'\{.*?\}', response_text, re.DOTALL)

    answer_dict = None
    for json_str in json_matches:
        try:
            answer_dict = json.loads(json_str)
            break
        except json.JSONDecodeError:
            continue

    ret = dict(data)
    ret['pred'] = answer_dict
    return ret








def run_hallucination_study(raw_dataset="./Hallucination_study/dataset/test.jsonl",
                            output_file="./Hallucination_study/dataset/prediction.jsonl",
                            model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"):
    """
    Runs a full hallucination detection pipeline using a specified LLM on the RAGTruth dataset.

    Args:
        raw_dataset (str, optional): Path to the input JSONL file containing hallucination evaluation data.
                                     Default is './Hallucination_study/dataset/test.jsonl'.
        output_file (str, optional): Path to save the prediction results. Default is './Hallucination_study/dataset/prediction.jsonl'.
        model_id (str, optional): Hugging Face model identifier for the hallucination judge. Default is 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit'.

    Returns:
        None: Outputs model predictions to a JSONL file and prints evaluation metrics to the console.

    Process:
        1. Checks for existence of required dataset splits (train, dev, test); generates them if missing.
        2. Loads the evaluation data from the specified test split file.
        3. Initializes the LLM hallucination detection model via Hugging Face pipeline.
        4. Iterates through all samples and generates hallucination predictions using the model.
        5. Writes all predictions to the specified output file in JSONL format.
        6. Computes and prints hallucination detection metrics (Accuracy, Precision, Recall, F1)
           across the entire dataset and per task type (QA, Summary, Data2txt).
    """
    
    # Step 1: Check if dataset files exist; if not, generate them
    required_files = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    dataset_dir = "./Hallucination_study/dataset"
    missing_files = [
        f for f in required_files
        if not os.path.exists(os.path.join(dataset_dir, f))
    ]

    if missing_files:
        print(f"Missing files: {missing_files}. Running get_data() to generate them...")
        get_data()

    # Step 2: Load task data
    tasks = []
    with open(raw_dataset, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading tasks"):
            data = json.loads(line)
            tasks.append(data)

    print(f"Total tasks: {len(tasks)}")

    # Step 3: Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading tokenizer and model...")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    # Step 4: Generate responses
    results = []
    pbar = tqdm(total=len(tasks), desc="Generating responses")
    for data in tasks:
        try:
            answer = generate_response(data, pipeline)
            results.append(answer)
        except Exception as e:
            print(f"Error generating response: {e}")
            results.append({"error": str(e)})
        pbar.update(1)
    pbar.close()

    # Step 5: Save predictions to output file
    bad_sample = 0
    with open(output_file, 'w') as f:
        for d in results:
            if isinstance(d, dict):
                f.write(json.dumps(d) + "\n")
            else:
                bad_sample += 1
                print(d)
    print(f"Bad samples: {bad_sample}")

    # Step 6: Evaluation
    df = pd.DataFrame.from_records(results)
    df['is_halu'] = df['labels'].apply(lambda x: len(x) > 0)
    df['pred_halu'] = df['pred'].apply(
        lambda x: (len(x.get('hallucination list', [])) > 0) if x is not None else False
    )

    print(f"\nTotal number of samples: {len(df)}")
    print_metrics(df)

    for task in ['QA', 'Summary', 'Data2txt']:
        temp = df[df['task_type'] == task]
        print(f'\n{task} - Number of samples: {len(temp)}')
        print_metrics(temp)


def print_metrics(df):
    print(
        f"Accuracy: {accuracy_score(df['is_halu'], df['pred_halu']):.3f}\t"
        f"Precision: {precision_score(df['is_halu'], df['pred_halu']):.3f}\t"
        f"Recall: {recall_score(df['is_halu'], df['pred_halu']):.3f}\t"
        f"F1_score: {f1_score(df['is_halu'], df['pred_halu']):.3f}"
    )

