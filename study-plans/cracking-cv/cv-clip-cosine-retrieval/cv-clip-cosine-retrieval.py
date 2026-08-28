import numpy as np
def cosine_topk(queries, corpus, k):
    """
    Returns: dict with keys 'indices' (M x k ints) and 'scores' (M x k floats rounded to 4 decimals)
    """
    eps = 1e-12

    q_norms = np.linalg.norm(queries, axis = 1, keepdims = True)
    c_norms = np.linalg.norm(corpus, axis = 1, keepdims= True)

    q = queries / (q_norms + eps)
    c = corpus / (c_norms + eps)

    similarities = q @ c.T # M x N

    M,N = similarities.shape
    
    top_indices = np.empty((M,k), dtype = np.int64)
    top_scores = np.empty((M,k), dtype = float)

    corpus_indices = np.arange(N)

    for i in range(M):
        order = np.lexsort(
            (corpus_indices, -similarities[i])
        )

        selected = order[:k]

        top_indices[i] = selected
        top_scores[i] = similarities[i, selected]

    top_scores = np.round(top_scores, 4)
    return {
        "indices": top_indices,
        "scores": top_scores,
    }
