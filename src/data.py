from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_20261020_161901.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_20261020_161901.csv"

def load_split_data():
    """Load the existing fixed train/test split"""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test

def prepare_model_data(train, test):
    """Create ID mapping and prepare model-ready data"""
    user_ids = np.sort(train["userId"].unique())
    movie_ids = np.sort(train["movieId"].unique())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    train_model = train.copy()
    test_model = test.copy()

    train_model["user_idx"] = train_model["userId"].map(user_to_idx)
    train_model["movie_idx"] = train_model["movieId"].map(movie_to_idx)

    test_model["user_idx"] = test_model["userId"].map(user_to_idx)
    test_model["movie_idx"] = test_model["movieId"].map(movie_to_idx)

    test_known = test_model.dropna(subset=["user_idx", "movie_idx"]).copy()

    train_model["user_idx"] = train_model["user_idx"].astype("int64")
    train_model["movie_idx"] = train_model["movie_idx"].astype("int64")
    test_known["user_idx"] = test_known["user_idx"].astype("int64")
    test_known["movie_idx"] = test_known["movie_idx"].astype("int64")

    catalog = pd.DataFrame({
        "movieId": movie_ids,
        "movie_idx": np.arange(len(movie_ids), dtype=np.int64),
    })

    return {
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "train_model": train_model,
        "test_known": test_known,
        "catalog": catalog,
    }

def chronological_validation_split(train_model, val_fraction=0.1):
    """Split each user's interactions chronologically"""
    fit_parts = []
    val_parts = []

    for _, group in train_model.groupby("user_idx", sort=False):
        group = group.sort_values("timestamp")
        split_idx = int(len(group) * (1 - val_fraction))

        if len(group) >= 3:
            fit_parts.append(group.iloc[:split_idx])
            val_parts.append(group.iloc[split_idx:])
        else:
            fit_parts.append(group)

    fit_df = pd.concat(fit_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    val_positive_df = val_df[val_df["rating"] >= 4].copy()

    return fit_df, val_df, val_positive_df


def generate_bpr_pairs(pos_df, pos_movies_by_user, all_movie_indices):
    """Generate one negative item for each positive interaction"""
    rng = np.random.default_rng(42)

    users = pos_df["user_idx"].to_numpy(dtype=np.int64)
    pos_movies = pos_df["movie_idx"].to_numpy(dtype=np.int64)
    neg_movies = np.empty(len(pos_df), dtype=np.int64)

    for i, user_idx in enumerate(users):
        pos_set = pos_movies_by_user[user_idx]

        while True:
            neg_movie = rng.choice(all_movie_indices)
            if neg_movie not in pos_set:
                neg_movies[i] = neg_movie
                break

    return users, pos_movies, neg_movies