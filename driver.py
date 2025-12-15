from data_io import load_ratings, load_items
from encoding import build_id_maps, add_index_columns
from cf import build_user_item_matrix, fit_svd, recommend_cf_top_n
from cb import fit_tfidf_item_profiles, recommend_similar_items
from hybrid import recommend_hybrid_top_n


if __name__ == "__main__":
    rating_path = "data/Amazon_CDs_and_Vinyl.inter"
    item_path = "data/Amazon_CDs_and_Vinyl.item"

    ratings_df = load_ratings(rating_path)
    items_df = load_items(item_path)

    # Build reproducible mappings from ratings data
    user2idx, item2idx, idx2user, idx2item = build_id_maps(ratings_df,
                                                           sort_ids=True)

    # Add index columns
    ratings_idx = add_index_columns(ratings_df, user2idx, item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)

    # Sparse matrix
    user_item = build_user_item_matrix(ratings_idx, n_users, n_items)

    # CF model
    svd, U, V = fit_svd(user_item, n_factors=50, random_state=42)

    # TF-IDF content
    tfidf, item_profiles = fit_tfidf_item_profiles(items_df,
                                                   text_col="categories")

    # Demo recommendations
    print("CF recs:", recommend_cf_top_n(user_idx=0, U=U, V=V,
                                         user_item_matrix=user_item,
                                         idx2item=idx2item, n=5))

    # pick any item_idx that exists in CF universe (0..n_items-1)
    print("Similar items:", recommend_similar_items(
        item_idx=0, item_profiles=item_profiles, idx2item=idx2item, n=5))

    print("Hybrid recs:", recommend_hybrid_top_n(
        user_idx=0, U=U, V=V, user_item_matrix=user_item,
        item_profiles=item_profiles, idx2item=idx2item, n=5, alpha=0.7))
