import numpy as np


def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
    return relevant_retrieved / k

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0  # Avoid division by zero
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
    return relevant_retrieved / len(relevant)

def dcg_at_k(retrieved, relevant, k):
    dcg = sum([(1 / np.log2(rank + 1)) if doc in relevant else 0 
               for rank, doc in enumerate(retrieved[:k], start=1)])
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_dcg = dcg_at_k(relevant, relevant, k)  # Best possible DCG
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

