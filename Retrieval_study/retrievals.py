from sentence_transformers import util
import torch






device = "cuda" if torch.cuda.is_available() else "cpu"  # Ensure device consistency



def retrieve_with_e5(model, question, docs, k=5):
    """
    Retrieves top-k documents using the E5 embedding model based on cosine similarity.

    Args:
        model (SentenceTransformer): Preloaded E5 sentence embedding model.
        question (str): The query/question string.
        docs (list of tuples): List of document tuples where each item is (doc_id, doc_text).
        k (int): Number of top documents to retrieve.

    Returns:
        list: List of top-k document identifiers ranked by similarity to the query.

    Process:
        - Encodes the question into an embedding using the E5 model.
        - Encodes the document texts into embeddings.
        - Computes cosine similarity between the question and all documents.
        - Selects the indices of the top-k most similar documents.
        - Returns the corresponding document IDs.
    """
    
    question_emb = model.encode(question, convert_to_tensor=True)
    doc_embs = model.encode([d[1] for d in docs], convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
    top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
    return [docs[i][0] for i in top_k_indices]




def retrieve_with_contriever(model, tokenizer, question, docs, k=5):
    """
    Retrieves top-k documents using the Contriever model based on cosine similarity.

    Args:
        model (transformers.PreTrainedModel): Preloaded Contriever encoder model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the Contriever model.
        question (str): Input query string.
        docs (list of tuples): List of document tuples in the form (doc_id, doc_text).
        k (int): Number of top documents to retrieve.

    Returns:
        list: List of top-k document identifiers ranked by similarity to the query.

    Process:
        - Tokenizes and encodes the query into a hidden-state embedding.
        - Batch tokenizes and encodes all document texts into embeddings.
        - Computes cosine similarity between query and document embeddings.
        - Sorts and selects the top-k highest scoring document indices.
        - Returns the corresponding document IDs.
    """

    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        question_emb = model(**inputs).last_hidden_state.mean(dim=1)
    
    # Tokenize all documents at once (batch processing)
    doc_texts = [doc[1] for doc in docs]  # Extract text from docs
    doc_inputs = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        doc_embs = model(**doc_inputs).last_hidden_state.mean(dim=1)
    
    scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
    top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
    return [docs[i][0] for i in top_k_indices]




def retrieve_with_finbert(model, tokenizer, question, docs, k=5):
    """
    Retrieves top-k documents using the FinBERT model based on cosine similarity.

    Args:
        model (transformers.PreTrainedModel): Preloaded FinBERT encoder model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the FinBERT model.
        question (str): The input financial query.
        docs (list of tuples): List of document tuples in the format (doc_id, doc_text).
        k (int): Number of top documents to retrieve.

    Returns:
        list: List of top-k document identifiers ranked by semantic similarity to the question.

    Process:
        - Tokenizes and encodes the question into a dense embedding using FinBERT.
        - Batch tokenizes and encodes all document texts into embeddings.
        - Computes cosine similarity between the query and each document.
        - Identifies and returns the IDs of the top-k most similar documents.
    """

    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        question_emb = model(**inputs).last_hidden_state.mean(dim=1)
    
    # Tokenize all documents at once (batch processing)
    doc_texts = [doc[1] for doc in docs]  # Extract text from docs
    doc_inputs = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        doc_embs = model(**doc_inputs).last_hidden_state.mean(dim=1)
    
    scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
    top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
    return [docs[i][0] for i in top_k_indices]




def retrieve_with_fingpt(model, tokenizer, question, docs, k=5):
    """
    Retrieves top-k documents using the FinGPT model based on cosine similarity from hidden state embeddings.

    Args:
        model (transformers.PreTrainedModel or PeftModel): LoRA-adapted FinGPT model for financial tasks.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the FinGPT base model.
        question (str): The financial query to retrieve relevant information for.
        docs (list of tuples): List of document entries as (doc_id, doc_text).
        k (int): Number of top documents to retrieve.

    Returns:
        list: List of top-k document identifiers based on semantic similarity to the query.

    Process:
        - Tokenizes and encodes the input query, extracting the last hidden state as the question embedding.
        - Tokenizes and encodes all document texts in a batch, extracting their last hidden states as embeddings.
        - Computes cosine similarity between query embedding and each document embedding.
        - Selects the top-k highest scoring documents and returns their IDs.
    """

    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        question_emb = outputs.hidden_states[-1].mean(dim=1).to(device)
    
    # Tokenize all documents at once (batch processing)
    doc_texts = [doc[1] for doc in docs]  # Extract text from docs
    doc_inputs = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        doc_outputs = model(**doc_inputs, output_hidden_states=True)  
        doc_embs = doc_outputs.hidden_states[-1].mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
    scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
    top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
    return [docs[i][0] for i in top_k_indices]
