from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def fit_tfidf_item_profiles(items_df,
                            text_col="categories",
                            token_pattern=r"[^,]+"):
    tfidf = TfidfVectorizer(token_pattern=token_pattern)
    item_profiles = tfidf.fit_transform(items_df[text_col].fillna(""))
    return tfidf, item_profiles


def recommend_similar_items(item_idx, item_profiles, idx2item, n=5):
    sims = cosine_similarity(item_profiles[item_idx], item_profiles).ravel()
    top = np.argsort(-sims)
    top = top[top != item_idx]  # exclude itself
    top = top[:n]
    return [idx2item[i] for i in top]
