import numpy as np


def precision_at_k(retrieved, relevant, k):
    """
    Computes precision at rank k for a list of retrieved items.

    Args:
        retrieved (list): A ranked list of retrieved item identifiers.
        relevant (list or set): A set of ground truth relevant item identifiers.
        k (int): The number of top items to consider for evaluation.

    Returns:
        float: Precision score at rank k (i.e., fraction of top-k items that are relevant).

    Process:
        - Truncates the retrieved list to its top-k elements.
        - Computes the number of items in the top-k that are also in the relevant set.
        - Divides the count of relevant retrieved items by k to compute precision.
    """

    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
    return relevant_retrieved / k




def recall_at_k(retrieved, relevant, k):
    """
    Computes recall at rank k for a list of retrieved items.

    Args:
        retrieved (list): A ranked list of retrieved item identifiers.
        relevant (list or set): A set of ground truth relevant item identifiers.
        k (int): The number of top items to consider for evaluation.

    Returns:
        float: Recall score at rank k (i.e., fraction of relevant items that appear in the top-k retrieved items).

    Process:
        - Returns 0 if there are no relevant items (to avoid division by zero).
        - Truncates the retrieved list to the top-k items.
        - Counts how many of these top-k items are present in the relevant set.
        - Divides this count by the total number of relevant items to compute recall.
    """
    
    if not relevant:
        return 0  # Avoid division by zero
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
    return relevant_retrieved / len(relevant)




def dcg_at_k(retrieved, relevant, k):
    """
    Computes the Discounted Cumulative Gain (DCG) at rank k.

    Args:
        retrieved (list): A ranked list of retrieved item identifiers.
        relevant (list or set): A set of ground truth relevant item identifiers.
        k (int): The number of top items to consider for evaluation.

    Returns:
        float: DCG score at rank k, reflecting the usefulness of a document based on its position.

    Process:
        - Iterates over the top-k retrieved items with their ranks.
        - For each relevant item, adds a gain of 1 divided by log2 of the rank (starting at 1).
        - Irrelevant items contribute 0 to the score.
        - Sums the discounted gains to compute the final DCG score.
    """

    dcg = sum([(1 / np.log2(rank + 1)) if doc in relevant else 0 
               for rank, doc in enumerate(retrieved[:k], start=1)])
    return dcg




def ndcg_at_k(retrieved, relevant, k):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        retrieved (list): A ranked list of retrieved item identifiers.
        relevant (list or set): A set of ground truth relevant item identifiers.
        k (int): The number of top items to consider for evaluation.

    Returns:
        float: NDCG score at rank k, normalized between 0 and 1.

    Process:
        - Computes the actual DCG for the top-k retrieved items.
        - Computes the ideal DCG using the ideal ranking (i.e., relevant items ranked at the top).
        - Divides the actual DCG by the ideal DCG to obtain the normalized score.
        - Returns 0 if the ideal DCG is zero (to avoid division by zero).
    """

    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_dcg = dcg_at_k(relevant, relevant, k)  # Best possible DCG
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

