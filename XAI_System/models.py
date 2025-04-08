import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import pipeline
import transformers
import json
from sklearn import svm


device = "cuda" if torch.cuda.is_available() else "cpu"




def retrieve_with_contriever(question, chunks, k, approach='cosine'):

    print('START RETRIEVAL:.....................')
    # Load retrieval models
    contriever_model = AutoModel.from_pretrained("facebook/contriever").to(device)
    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    # Embed question
    inputs = contriever_tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        question_emb = contriever_model(**inputs).last_hidden_state.mean(dim=1)
    
    # Tokenize all documents at once (batch processing)
    doc_inputs = contriever_tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        doc_embs = contriever_model(**doc_inputs).last_hidden_state.mean(dim=1)
    
    if approach=='cosine':
        scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
        top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
        retrieved = [(chunks[i], float(scores[i])) for i in top_k_indices]
        return retrieved
    elif approach=='svm':
        x = np.concatenate([question_emb.cpu().detach().numpy(), doc_embs.cpu().detach().numpy()])
        y = np.zeros(doc_embs.size(0) + 1)
        y[0] = 1 # we have a single positive example

        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1, dual="auto")
        clf.fit(x, y)
        similarities = clf.decision_function(x)
        top_k_indices = np.argsort(-similarities)[1:k+1]
        retrieved = [(chunks[i-1], float(similarities[i])) for i in top_k_indices]
        return retrieved
    else:
        raise ValueError(f"Invalid Approach: {approach}") 





def generate_llm_response(query, contexts):
    print('LOADING GENERATOR MODEL:.....................')
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    context_text = "\n".join(contexts)

    prompt = f"""
    You are a financial assistant. Provide a short, fact-based answer strictly using the provided context. 
    Do NOT explain your reasoning. Do NOT include reflections or intermediate thoughts. Output ONLY the final answer.


    Question: {query}

    Context:
    {context_text}

    Answer:"""

    result = generator(prompt, max_new_tokens=1024)
    generated_text = result[0]["generated_text"]
    answer = generated_text.split("Answer:")[-1].split("</think>")[-1].strip()
    return answer








def process_dialog_to_single_turn(data):
    TEMPLATE=(
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
        )

    prompt = TEMPLATE.format(
        question=data['question'], 
        reference=data['reference'], 
        response=data['response']
    )   
    return prompt





def detect_hallucination(data):
    print("Loading LLM as a judge model............")
    model_id =   "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"       

    llm_judge = pipeline(
        "text-generation",
        model = model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    input_prompt = process_dialog_to_single_turn(data)
    messages = [
        {"role": "system", "content": "You are a helpful assistant!"},
        {"role": "user", "content": input_prompt},
    ]

    outputs = llm_judge(messages, max_new_tokens=1024)

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

    return answer_dict







def compare_semantic_similarity(embedder, text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return float(similarity[0][0])


def analyze_context_influence(query, contexts, base_response):
    # Load SBERT model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, effective

    scores = []
    for i in range(len(contexts)):
        reduced_context = contexts[:i] + contexts[i+1:]
        response = generate_llm_response(query, reduced_context)
        similarity = compare_semantic_similarity(embedder, base_response, response)
        scores.append((i, similarity))

        print(f"\n[Without Context {i+1}] Semantic Similarity between base and new response: {similarity:.4f}")

    return scores







def extract_rationale(query, contexts, answer):

    # Ensure contexts are strings
    if isinstance(contexts[0], tuple):
        contexts = [c[0] for c in contexts]

    context_text = "\n".join(contexts)

    rationale_prompt = f"""
    You are an assistant tasked with identifying which parts of the provided context were most important in forming a given answer.

    Instructions:
    - ONLY return the exact spans or sentences from the context that directly supported the answer.
    - Answer must contains ONLY elements from Context
    - DO NOT explain your reasoning.
    - DO NOT include anything that was not used to justify the answer.
    - DO NOT rephrase or summarize.
    - Format the output as a bullet list of the original context segments.

    Question:
    {query}

    Context:
    {context_text}

    Answer:
    {answer}

    Important supporting context:
    - 
    """

    # Run generation with strict prompt
    generator = pipeline(
        "text-generation",
        model= "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    result = generator(rationale_prompt, max_new_tokens=512)
    generated_text = result[0]["generated_text"]

    # Return the portion after the last instruction marker
    rationale_output = generated_text.split("Important supporting context:")[-1].split("</think>")[-1].strip()

    return rationale_output
