from sentence_transformers import util
import torch






device = "cuda" if torch.cuda.is_available() else "cpu"  # Ensure device consistency



def retrieve_with_e5(model, question, docs, k=5):
    
    question_emb = model.encode(question, convert_to_tensor=True)
    doc_embs = model.encode([d[1] for d in docs], convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, doc_embs)[0]
    top_k_indices = torch.argsort(scores, descending=True)[:k].tolist()
    return [docs[i][0] for i in top_k_indices]



def retrieve_with_contriever(model, tokenizer, question, docs, k=5):

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
