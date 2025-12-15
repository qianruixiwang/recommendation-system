import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_hybrid_top_n(user_idx, U, V, user_item_matrix,
                           item_profiles, idx2item, n=5, alpha=0.7):
    cf_scores = U[user_idx] @ V.T

    interacted = user_item_matrix[user_idx].indices
    if interacted.size > 0:
        # mean similarity of each item to the user's interacted items
        cb_scores = cosine_similarity(item_profiles,
                                      item_profiles[interacted]).mean(axis=1)
    else:
        cb_scores = np.zeros_like(cf_scores)

    hybrid = alpha * cf_scores + (1 - alpha) * cb_scores
    hybrid[interacted] = -np.inf
    top = np.argsort(-hybrid)[:n]
    return [idx2item[i] for i in top]
