from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np


def build_user_item_matrix(df, n_users, n_items,
                           user_idx_col="user_idx", item_idx_col="item_idx",
                           rating_col="rating"):
    return csr_matrix(
        (df[rating_col].astype(float), (df[user_idx_col], df[item_idx_col])),
        shape=(n_users, n_items)
    )


def fit_svd(user_item_matrix, n_factors=50, random_state=42):
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    U = svd.fit_transform(user_item_matrix)     # (n_users, k)
    V = svd.components_.T                       # (n_items, k)
    return svd, U, V


def recommend_cf_top_n(user_idx, U, V, user_item_matrix, idx2item, n=5):
    scores = U[user_idx] @ V.T
    interacted = user_item_matrix[user_idx].indices
    scores[interacted] = -np.inf
    top = np.argsort(-scores)[:n]
    return [idx2item[i] for i in top]
