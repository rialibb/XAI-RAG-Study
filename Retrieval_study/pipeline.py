import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from metrics import recall_at_k, ndcg_at_k
from retrievals import retrieve_with_e5, retrieve_with_contriever, retrieve_with_finbert, retrieve_with_fingpt


device = "cuda" if torch.cuda.is_available() else "cpu"  # Ensure device consistency




def import_models(retrievals):
    
    models = {}
    tokenizers = {}
    
    if 'E5' in retrievals:
        print('UPLOAD E5 MODEL________________')
        models['E5'] = SentenceTransformer("intfloat/e5-large").to(device)
        
    if 'Contriever' in retrievals:
        print('UPLOAD Contriever MODEL________')
        models['Contriever'] = AutoModel.from_pretrained("facebook/contriever").to(device)
        tokenizers['Contriever'] = AutoTokenizer.from_pretrained("facebook/contriever")
    
    if 'FinBERT' in retrievals:
        print('UPLOAD FinBERT MODEL___________')
        models['FinBERT'] = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
        tokenizers['FinBERT'] = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        
        
    if 'FinGPT' in retrievals:
        print('UPLOAD FinGPT MODEL____________')
        base_model_name = 'NousResearch/Llama-2-7b-hf'
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16).to(device)
        tokenizers['FinGPT'] = AutoTokenizer.from_pretrained(base_model_name)
        adapter_model_name = 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'
        models['FinGPT'] = PeftModel.from_pretrained(
            model=base_model,
            model_id=adapter_model_name,  
            adapter_name="default", 
            device_map="auto").to(device)
        models['FinGPT'].eval()
        
    return models, tokenizers
        
        
        
        
        
        
        
    

def run_retrieval_study(DATASET_TYPE='tatqa',   # 'finqa' or 'tatqa'
                        retrievals=['E5', 'Contriever', 'FinBERT', 'FinGPT'],    # one or more of the four
                        k=5):   

    # Load the dataset
    df = pd.read_parquet(f"hf://datasets/rungalileo/ragbench/{DATASET_TYPE}/test-00000-of-00001.parquet")
    df = df[['documents_sentences', 'question', 'all_relevant_sentence_keys']].reset_index(drop=True)
    print('FINISH IMPORTING DATA')
    
    # Preprocessing
    for i in range(len(df)):
        arrs = df.loc[i, 'all_relevant_sentence_keys']
        df.loc[i, 'all_relevant_sentence_keys'] = [s.rstrip('.') for s in arrs]

        all_lists = []
        for big_l in df.loc[i, 'documents_sentences']:
            all_lists.extend(big_l)
        df.loc[i, 'documents_sentences'] = all_lists

    # import models 
    models, tokenizers = import_models(retrievals)
    
    # Step 1: Use defaultdict to initialize results
    results = defaultdict(lambda: {'recall': [], 'ndcg': []})

    # Step 2: Retrieval loop
    for i, row in tqdm(df.iterrows(), total=len(df)):

        question = row['question']
        docs = row['documents_sentences']
        relevant_docs = row['all_relevant_sentence_keys']
        
        if 'E5' in retrievals:
            retrieved = retrieve_with_e5(models['E5'], question, docs, k)
            results['E5']['recall'].append(recall_at_k(retrieved, relevant_docs, k))
            results['E5']['ndcg'].append(ndcg_at_k(retrieved, relevant_docs, k))
        
        if 'Contriever' in retrievals:
            retrieved = retrieve_with_contriever(models['Contriever'], tokenizers['Contriever'], question, docs, k)
            results['Contriever']['recall'].append(recall_at_k(retrieved, relevant_docs, k))
            results['Contriever']['ndcg'].append(ndcg_at_k(retrieved, relevant_docs, k))

        if 'FinBERT' in retrievals:
            retrieved = retrieve_with_finbert(models['FinBERT'], tokenizers['FinBERT'], question, docs, k)
            results['FinBERT']['recall'].append(recall_at_k(retrieved, relevant_docs, k))
            results['FinBERT']['ndcg'].append(ndcg_at_k(retrieved, relevant_docs, k))

        if 'FinGPT' in retrievals:
            retrieved = retrieve_with_fingpt(models['FinGPT'], tokenizers['FinGPT'], question, docs, k)
            results['FinGPT']['recall'].append(recall_at_k(retrieved, relevant_docs, k))
            results['FinGPT']['ndcg'].append(ndcg_at_k(retrieved, relevant_docs, k))

    # Step 3: Evaluation
    for retrieval in retrievals:
        print(f'For {retrieval} _____________')
        print(f"Recall = {np.mean(results[retrieval]['recall']).item():.4f}")
        print(f"NDCG = {np.mean(results[retrieval]['ndcg']).item():.4f}")
        print('\n')



run_retrieval_study()