import numpy as np
import pandas as pd
import torch

from src.data import load_split_data, prepare_model_data
from src.model import MatrixFactorization


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bpr_model.pt"
REPORTS_DIR = PROJECT_ROOT / "reports"

def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return np.nan
    return len(set(recommended[:k]) & relevant) / len(relevant)

def ndcg_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    relevance = np.array([1.0 if movie in relevant else 0.0 for movie in recommended])
    discounts = 1 / np.log2(np.arange(2, len(relevance) + 2))
    dcg = np.sum(relevance * discounts)

    ideal_length = min(len(relevant), k)
    idcg = np.sum(1 / np.log2(np.arange(2, ideal_length + 2)))

    return float(dcg / idcg) if idcg > 0 else 0.0

def evaluate_recommendations(rank_function, seen_by_user, relevant_by_user, k=10):
    recalls, ndcgs = [], []

    for user, relevant in relevant_by_user.items():
        if not relevant:
            continue

        seen = seen_by_user.get(user, set())
        recommendations = rank_function(user, seen, k)

        recalls.append(recall_at_k(recommendations, relevant, k))
        ndcgs.append(ndcg_at_k(recommendations, relevant, k))

    return {
        "Recall@10": float(np.nanmean(recalls)) if recalls else 0.0,
        "NDCG@10": float(np.nanmean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": len(recalls)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test = load_split_data()
    data = prepare_model_data(train, test)

    train_model = data["train_model"]
    test_known = data["test_known"]
    n_users = len(data["user_ids"])
    n_movies = len(data["movie_ids"])

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True,)

    model = MatrixFactorization(
        n_users=checkpoint["n_users"],
        n_movies=checkpoint["n_movies"],
        emb_dim=checkpoint["embedding_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    seen_by_user = (train_model.groupby("user_idx")["movie_idx"].apply(set).to_dict())

    test_pos = test_known.loc[test_known["rating"] >= 4]
    relevant_by_user = (test_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict())

    # Exclude any relevant movies already seen in training.
    relevant_by_user = {
        user: relevant - seen_by_user.get(user, set())
        for user, relevant in relevant_by_user.items()
    }
    relevant_by_user = {
        user: relevant
        for user, relevant in relevant_by_user.items()
        if relevant
    }

    all_movies = np.arange(n_movies, dtype=np.int64)

    def bpr_rank(user, seen, k):
        candidates = np.array([movie for movie in all_movies if movie not in seen], dtype=np.int64)
        if len(candidates) == 0:
            return []

        user_tensor = torch.full((len(candidates),), user, dtype=torch.long, device=device)
        movie_tensor = torch.as_tensor(candidates, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model.score(user_tensor, movie_tensor)

        top_indices = torch.topk(scores, k=min(k, len(candidates))).indices
        return candidates[top_indices.cpu().numpy()].tolist()

    # Popularity = number of training interactions per movie.
    popularity = (
        train_model.groupby("movie_idx")
        .size()
        .reindex(all_movies, fill_value=0)
        .to_numpy()
    )
    popular_movies = np.argsort(-popularity)

    def popularity_rank(user, seen, k):
        recommendations = [int(movie) for movie in popular_movies if movie not in seen]
        return recommendations[:k]

    bpr_metrics = evaluate_recommendations(bpr_rank, seen_by_user, relevant_by_user)
    popularity_metrics = evaluate_recommendations(popularity_rank, seen_by_user, relevant_by_user)

    results = pd.DataFrame([
        {"model": "Popularity baseline", **popularity_metrics},
        {"model": "Matrix Factorization (BPR)", **bpr_metrics}
    ])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "test_metrics.csv", index=False)

    print("\nTest results:")
    print(results.to_string(index=False))
    print(f"\nMetrics saved to: {REPORTS_DIR / 'test_metrics.csv'}")

if __name__ == "__main__":
    main()
