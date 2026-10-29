import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from src.data import load_split_data, prepare_model_data
from src.model import MatrixFactorization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bpr_model.pt"
INDEX_PATH = PROJECT_ROOT / "models" / "movie_index.faiss"
MOVIE_IDS_PATH = PROJECT_ROOT / "models" / "movie_index_ids.npy"
MOVIES_PATH = PROJECT_ROOT / "data" / "raw" / "movies.csv"


def recommend(user_id: int, top_k: int = 10):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True,)

    user_ids = np.asarray(checkpoint["user_ids"], dtype=np.int64)
    user_to_idx = {int(original_id): idx for idx, original_id in enumerate(user_ids)}

    if user_id not in user_to_idx:
        raise ValueError(
            f"Unknown userId={user_id}. "
            "Only users present in the training data are supported."
        )

    user_idx = user_to_idx[user_id]

    # Load training data to identify movies the user has already rated.
    train, _ = load_split_data()

    seen_movies = set(train.loc[train["userId"] == user_idx, "movieId"].astype(int))

    # Load FAISS index and the mapping from index positions to movie IDs.
    index = faiss.read_index(str(INDEX_PATH))
    movie_ids = np.load(MOVIE_IDS_PATH)

    # Load model to get the user's embedding.
    model = MatrixFactorization(
        n_users=checkpoint["n_users"],
        n_movies=checkpoint["n_movies"],
        emb_dim=checkpoint["embedding_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        user_vector = (model.user_emb.weight[user_idx].detach().cpu().numpy())

    query = np.append(user_vector, 1.0).astype(np.float32)

    # Search the whole catalog so filtering seen movies cannot
    # leave us with fewer than top_k recommendations.
    _, positions = index.search(query.reshape(1, -1), index.ntotal)
    ranked_movie_ids = movie_ids[positions[0]]

    recommendations = [
        int(movie_id)
        for movie_id in ranked_movie_ids
        if int(movie_id) not in seen_movies
    ][:top_k]

    movies = pd.read_csv(MOVIES_PATH)
    movie_lookup = movies.set_index("movieId")["title"]

    result = pd.DataFrame({
        "rank": range(1, len(recommendations) + 1),
        "movieId": recommendations,
        "title": [movie_lookup.get(movie_id, "Unknown title") for movie_id in recommendations]
    })

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Recommend movies using BPR Matrix Factorization and FAISS."
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="Original MovieLens userId."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to return."
    )
    args = parser.parse_args()

    result = recommend(args.user_id, args.top_k)

    print(f"\nRecommendations for userId={args.user_id}:\n")
    if result.empty:
        print("No recommendations available.")
    else:
        print(result.to_string(index=False))

if __name__ == "__main__":
    main()