import pandas as pd
import random
import re
import json
from os import path
random.seed(2024)




def get_json_data(data):
    """
    Converts a DataFrame of annotated hallucination data into a structured JSON-compatible format.

    Args:
        data (pandas.DataFrame): DataFrame containing hallucination annotations and source information.

    Returns:
        list of dict: A list of dictionaries, each representing one example with formatted labels and references.

    Process:
        - Iterates over each row in the DataFrame and converts it to a dictionary.
        - Removes the 'prompt' field and adds a 'type' field with value 'response'.
        - Organizes hallucination labels into two categories: 'baseless info' and 'conflict'.
        - Based on task type (`QA`, `Summary`, or `Data2txt`), extracts the appropriate reference context.
        - Appends the reformatted example to the output list.
    """

    to_save = []
    for idx, row in data.iterrows():
        d = row.to_dict()
        d.pop('prompt')
        d['type'] = 'response'
        label = sorted(d['labels'], key=lambda x: x['start'])
        format_label = {'baseless info': [], 'conflict': []}
        for l in label:
            if l['label_type'].lower().find('baseless')>=0:
                format_label['baseless info'].append(l['text'])
            else:
                format_label['conflict'].append(l['text'])
        d['format_label'] = format_label
        if row['task_type']=='QA':
            d['reference'] = row['source_info']['passages']
            d['question'] = row['source_info']['question']
        elif row['task_type']=='Summary':
            d['reference'] = row['source_info']
        else:
            d['reference'] = f"{row['source_info']}"
        to_save.append(d)
    return to_save







def read_ragtruth_split(ragtruth_dir, split):
    """
    Loads and filters a specific split from the RAGTruth dataset, merging response and source information.

    Args:
        ragtruth_dir (str): Path to the directory containing RAGTruth files.
        split (str): Dataset split to load ('train', 'dev', or 'test').

    Returns:
        pandas.DataFrame: A merged DataFrame containing high-quality responses and their source context.

    Process:
        - Reads the 'response.jsonl' file and filters for entries with the specified split and quality set to 'good'.
        - Reads the 'source_info.jsonl' file containing reference contexts.
        - Merges both datasets on the shared `source_id` field.
        - Prints the shape of the resulting DataFrame and returns it.
    """
    
    resp = pd.read_json(path.join(ragtruth_dir, 'response.jsonl'), lines=True)
    test = resp[(resp['split']==split)&(resp['quality']=='good')]
    oc = pd.read_json(path.join(ragtruth_dir, 'source_info.jsonl'), lines=True)
    test = test.merge(oc, on='source_id')
    print(test.shape)
    return test









def get_data():
    """
    Processes and splits the RAGTruth dataset into train, dev, and test sets in JSONL format.

    Args:
        None

    Returns:
        None: Writes three files ('train.jsonl', 'dev.jsonl', 'test.jsonl') to the dataset directory.

    Process:
        - Loads the 'train' split from RAGTruth and performs stratified sampling by task type to create a dev set (50 examples per type).
        - Splits the data into training and development sets based on sampled `source_id`s.
        - Formats the data using `get_json_data()` to organize hallucination labels and references.
        - Saves the processed training and dev examples to JSONL files.
        - Loads, processes, and writes the 'test' split in the same format.
    """

    data = read_ragtruth_split('./Hallucination_study/dataset', 'train')
    dev_source_id = []
    for task in ['QA', 'Summary', 'Data2txt']:
        source_ids = data[data['task_type']==task]['source_id'].unique().tolist()
        dev_source_id.extend(random.sample(source_ids, 50))

    train = data[~data['source_id'].isin(dev_source_id)].reset_index(drop=True)
    
    dev = data[data['source_id'].isin(dev_source_id)]
    print(dev['task_type'].value_counts())
    train['fold'] = -1
    dev['fold']=-1
    train = get_json_data(train)
    dev = get_json_data(dev)
    with open(f'./Hallucination_study/dataset/train.jsonl', 'w') as f:
        for d in train:
            f.write(json.dumps(d)+"\n")

    with open(f'./Hallucination_study/dataset/dev.jsonl', 'w') as f:
        for d in dev:
            f.write(json.dumps(d)+"\n")

    test = read_ragtruth_split('./Hallucination_study/dataset', 'test')
    test = get_json_data(test)
    with open(f'./Hallucination_study/dataset/test.jsonl', 'w') as f:
        for d in test:
            f.write(json.dumps(d)+"\n")
    
