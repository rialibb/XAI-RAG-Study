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
    """
    Retrieves top-k relevant chunks using the Contriever model with either cosine similarity or an SVM-based approach.

    Args:
        question (str): The user query to retrieve relevant information for.
        chunks (list of str): List of document chunks (sentences or passages).
        k (int): Number of top chunks to retrieve.
        approach (str): Similarity method to use: 'cosine' (default) or 'svm'.

    Returns:
        list of tuples: List of top-k retrieved chunks and their corresponding similarity scores.

    Process:
        - Loads the pretrained Contriever model and tokenizer.
        - Encodes the question and all chunks into embeddings using mean pooling of hidden states.
        - If approach is 'cosine':
            - Computes cosine similarity between the query and all chunks.
            - Selects the top-k highest scoring chunks.
        - If approach is 'svm':
            - Concatenates the query and chunk embeddings as input to an SVM classifier.
            - Uses decision scores to rank and select the top-k most relevant chunks.
        - Returns a list of (chunk, score) tuples.
    """

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
    """
    Generates a short, fact-based answer to a financial query using a language model and provided context.

    Args:
        query (str): The financial question to be answered.
        contexts (list of str): List of retrieved document chunks used as context.

    Returns:
        str: The generated answer based strictly on the provided context.

    Process:
        - Loads the DeepSeek-Llama model using Hugging Face's `pipeline`.
        - Joins the context chunks into a single formatted string.
        - Constructs a strict instruction-based prompt to enforce factuality and prevent reasoning or hallucination.
        - Uses the language model to generate a response from the prompt.
        - Extracts and cleans the answer from the generated text.
    """

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
    """
    Constructs a hallucination detection prompt for a single-turn QA interaction.
    """
    
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
    """
    Uses an LLM to detect hallucinations in a generated response based on a given question and reference context.

    Args:
        data (dict): Dictionary containing:
            - 'question' (str): The input query.
            - 'reference' (str or list): The source context or passages.
            - 'response' (str): The generated answer to evaluate.

    Returns:
        dict or None: A dictionary with the key "hallucination list" containing detected spans,
                      or None if no valid JSON output is parsed.

    Process:
        - Loads a hallucination judgment model (Llama-3.3-70B-Instruct).
        - Converts the input into a hallucination detection prompt via `process_dialog_to_single_turn`.
        - Passes the prompt to the model using a system/user message format.
        - Extracts the hallucination output from the model’s response by locating and parsing JSON-like substrings.
        - Returns the first valid JSON dictionary found, if any.
    """

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
    """
    Computes the semantic similarity score between two text inputs using a sentence embedding model.

    Args:
        embedder (SentenceTransformer): Preloaded sentence embedding model (e.g., SBERT).
        text1 (str): The first text input.
        text2 (str): The second text input.

    Returns:
        float: Cosine similarity score between the two text embeddings (range: -1 to 1).

    Process:
        - Encodes both input texts into dense vector representations using the embedder.
        - Computes the cosine similarity between the two vectors using `pytorch_cos_sim`.
        - Extracts and returns the similarity score as a float.
    """
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return float(similarity[0][0])


def analyze_context_influence(query, contexts, base_response):
    """
    Performs ablation testing to analyze the influence of each context chunk on the generated response.

    Args:
        query (str): The original input question.
        contexts (list of str): List of context chunks used to generate the base response.
        base_response (str): The original response generated using all context chunks.

    Returns:
        list of tuples: Each tuple contains (index, similarity) where:
            - index: Index of the removed context chunk.
            - similarity: Semantic similarity between the new and original response.

    Process:
        - Loads a lightweight SBERT model for semantic similarity comparison.
        - Iteratively removes one context chunk at a time from the full set.
        - Regenerates the answer using the reduced context.
        - Computes the semantic similarity between the new response and the base response.
        - Returns a list of similarity scores to quantify the contribution of each context chunk.
    """

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
    """
    Extracts the most relevant context spans that justify the generated answer using an LLM prompt.

    Args:
        query (str): The original user question.
        contexts (list of str or tuples): List of context chunks used for answering the query.
        answer (str): The generated answer to be justified.

    Returns:
        str: A bullet-pointed list of context segments identified as rationale for the answer.

    Process:
        - Converts context tuples (if present) to plain strings.
        - Joins all context chunks into a single block of text.
        - Constructs a prompt instructing the model to extract only the essential supporting content.
        - Uses a strict instruction format to avoid reasoning or summarization.
        - Passes the prompt to the LLM (DeepSeek-Llama) for generation.
        - Extracts the rationale list from the output by parsing after the instruction marker.
    """

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
