import numpy as np


def build_id_maps(df, user_col="user_id", item_col="item_id", sort_ids=True):
    users = df[user_col].dropna().unique()
    items = df[item_col].dropna().unique()

    if sort_ids:
        users = np.sort(users)
        items = np.sort(items)

    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}

    return user2idx, item2idx, idx2user, idx2item


def add_index_columns(df, user2idx, item2idx,
                      user_col="user_id", item_col="item_id",
                      user_idx_col="user_idx", item_idx_col="item_idx",
                      strict=True):
    out = df.copy()
    out[user_idx_col] = out[user_col].map(user2idx)
    out[item_idx_col] = out[item_col].map(item2idx)

    if strict:
        if out[user_idx_col].isna().any():
            bad = out.loc[out[user_idx_col].isna(), user_col].unique()[:5]
            raise ValueError(f"Unknown user ids found (examples): {bad}")
        if out[item_idx_col].isna().any():
            bad = out.loc[out[item_idx_col].isna(), item_col].unique()[:5]
            raise ValueError(f"Unknown item ids found (examples): {bad}")

    out[user_idx_col] = out[user_idx_col].astype("int64")
    out[item_idx_col] = out[item_idx_col].astype("int64")
    return out
