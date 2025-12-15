import pandas as pd


def load_ratings(path, sep="\t", header=0,
                 user_col="user_id", item_col="item_id",
                 rating_col="rating", time_col="timestamp"):
    df = pd.read_csv(
        path, sep=sep,
        names=[user_col, item_col, rating_col, time_col],
        header=header
    )
    return df


def load_items(path, sep="\t", header=0,
               item_col="item_id", cat_col="categories",
               avg_col="average_rating",
               num_col="rating_number", price_col="price"):
    df = pd.read_csv(
        path, sep=sep,
        names=[item_col, cat_col, avg_col, num_col, price_col],
        header=header
    )
    df[cat_col] = df[cat_col].fillna("")
    return df
