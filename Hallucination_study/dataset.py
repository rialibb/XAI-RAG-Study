# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import json
import random
from datetime import datetime
import re
from torch.utils.data import Dataset


TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Below is an answer:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Summary": (
        "Below is the original news:\n" 
        "{reference}\n\n"
        "Below is a summary of the news:\n"
        "{response}\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Data2txt": (
        "Below is a structured data in the JSON format:\n"
        "{reference}\n\n"
        "Below is an overview article written in accordance with the structured data:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
}

B_INST, E_INST = "[INST]", "[/INST]"




def process_dialog(dialog, tokenizer, min_turn_idx=0, return_prompt=False):
    """
    Converts a multi-turn dialog into tokenized inputs and masked labels for supervised fine-tuning.

    Args:
        dialog (list of str): Alternating list of user prompts and assistant responses (e.g., [user, assistant, user, assistant, ...]).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for encoding the dialog turns.
        min_turn_idx (int, optional): Minimum turn index from which labels start being supervised. Earlier turns are masked. Default is 0.
        return_prompt (bool, optional): If True, returns only the reconstructed prompt string. Default is False.

    Returns:
        Union[Tuple[list, list], str]: 
            - If return_prompt is False, returns (inputs, labels) where:
                - inputs (list of int): Token IDs of the full dialog.
                - labels (list of int): Token labels with IGNORE_INDEX (-100) for masked tokens.
            - If return_prompt is True, returns the prompt string used in the final turn.

    Process:
        - Asserts the dialog is at least two entries and trims it to an even number of turns.
        - Iterates over each turn pair (user, assistant):
            - Constructs the formatted instruction pair with special tokens.
            - Encodes each turn to token IDs and prepares corresponding labels.
            - Applies label masking before `min_turn_idx` to exclude earlier history from supervision.
        - Truncates both inputs and labels to the model's max length.
        - Returns either the tokenized input-label pair or just the prompt, depending on `return_prompt`.
    """

    IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
    assert len(dialog)>=2
    dialog = dialog[:2*len(dialog)//2]
    inputs = []
    labels = []
    total_turns = len(dialog)//2
    prompt = ""
    for turn_idx in range(total_turns):
        cur_turn_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST} {dialog[turn_idx*2+1].strip()}"
        
        turn_input = [tokenizer.bos_token_id]+ \
                     tokenizer.encode(cur_turn_text, 
                                      add_special_tokens=False,
                                      truncation=False)+ \
                     [tokenizer.eos_token_id]
        if turn_idx>=min_turn_idx:
            cur_turn_only_input_text = f"{B_INST} {dialog[turn_idx*2].strip()} {E_INST}"
            turn_only_input = tokenizer.encode(cur_turn_only_input_text, 
                                            add_special_tokens=False,
                                            truncation=False)
            turn_label = turn_input.copy()
            input_len = len(turn_only_input)+1
            for i in range(input_len): # plus one for bos
                turn_label[i] = IGNORE_INDEX
            prompt += cur_turn_only_input_text
        else:
            # for single turn training, we need to mask all history
            turn_label = [IGNORE_INDEX]*len(turn_input)
            prompt += cur_turn_text
        inputs.extend(turn_input)
        labels.extend(turn_label)
    if return_prompt:
        return prompt
    assert len(inputs)==len(labels)
    inputs = inputs[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    return inputs, labels








def process_dialog_to_single_turn(data, tokenizer, return_prompt=False):
    """
    Prepares a single-turn hallucination detection prompt and optionally tokenizes it for model training.

    Args:
        data (dict): A dictionary containing:
            - 'task_type' (str): Task type, one of ['QA', 'Summary', 'Data2txt'].
            - 'question' (str): (Required for QA) The input question.
            - 'reference' (str or list): The context used to generate the response.
            - 'response' (str): The generated output to be evaluated.
            - 'labels' (list of dict): Hallucination annotations with 'text' and 'start' keys.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to convert text into model inputs.
        return_prompt (bool, optional): If True, returns only the raw prompt text. Default is False.

    Returns:
        Union[str, Tuple[list, list]]:
            - If `return_prompt=True`: returns the formatted prompt string.
            - If `return_prompt=False`: returns a tuple (inputs, labels) for training, with hallucination annotations.

    Process:
        - Constructs the hallucination detection prompt based on task type using pre-defined templates.
        - If only the prompt is needed, returns the formatted string.
        - Otherwise:
            - Extracts hallucinated spans from the labels.
            - Converts the prompt and JSON label to input-label token sequences using `process_dialog`.
    """

    if data['task_type']=='QA':
        prompt = TEMPLATES[data['task_type']].format(
            question=data['question'], 
            reference=data['reference'], 
            response=data['response']
        )
    else:
        prompt = TEMPLATES[data['task_type']].format(
            reference=data['reference'], 
            response=data['response']
        )        
    if return_prompt:
        return prompt
    label = sorted(data['labels'], key=lambda x: x['start'])
    label_dict = {
        'hallucination list': [x['text'] for x in label]
    }
    return process_dialog([prompt, json.dumps(label_dict, indent=2)], tokenizer)
    
