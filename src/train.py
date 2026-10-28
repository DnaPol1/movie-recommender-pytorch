import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    load_split_data,
    prepare_model_data,
    chronological_validation_split,
    generate_bpr_pairs
)
from src.model import MatrixFactorization

SEED = 42
EMB_DIM = 16
LR = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 1024

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
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

def evaluate_ranking(model, fit_df, val_pos_df, n_movies, device, k=10):
    seen_by_user = (fit_df.groupby("user_idx")["movie_idx"].apply(set).to_dict())
    relevant_by_user = (val_pos_df.groupby("user_idx")["movie_idx"].apply(set).to_dict())

    relevant_by_user = {
        user: relevant - seen_by_user.get(user, set())
        for user, relevant in relevant_by_user.items()
    }
    relevant_by_user = {
        user: relevant
        for user, relevant in relevant_by_user.items()
        if relevant
    }

    recalls, ndcgs = [], []
    all_movies = np.arange(n_movies)

    model.eval()
    with torch.no_grad():
        for user, relevant in relevant_by_user.items():
            seen = seen_by_user.get(user, set())
            candidates = np.array(
                [movie for movie in all_movies if movie not in seen],
                dtype=np.int64,
            )
            if len(candidates) == 0:
                continue

            user_tensor = torch.full(
                (len(candidates),), user, dtype=torch.long, device=device
            )
            movie_tensor = torch.as_tensor(candidates, dtype=torch.long, device=device)
            scores = model.score(user_tensor, movie_tensor)

            top_indices = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().numpy()
            recommendations = candidates[top_indices].tolist()

            recalls.append(recall_at_k(recommendations, relevant, k))
            ndcgs.append(ndcg_at_k(recommendations, relevant, k))

    return {
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": len(recalls),
    }

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train, test = load_split_data()
    data = prepare_model_data(train, test)

    train_model = data["train_model"]
    n_users = len(data["user_ids"])
    n_movies = len(data["movie_ids"])

    fit_df, _, val_pos_df = chronological_validation_split(train_model)

    # Generate training pairs using only positive interactions from fit_df.
    pos_df = fit_df.loc[fit_df["rating"] >= 4].copy()
    pos_movies_by_user = (pos_df.groupby("user_idx")["movie_idx"].apply(set).to_dict())
    all_movie_indices = np.arange(n_movies, dtype=np.int64)

    users, pos_movies, neg_movies = generate_bpr_pairs(
        pos_df,
        pos_movies_by_user,
        all_movie_indices
    )

    # Prevent leakage: evaluation history is fit_df, not all train_model.
    # Training pairs are also drawn only from fit_df.
    model = MatrixFactorization(n_users=n_users, n_movies=n_movies, emb_dim=EMB_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(
        torch.tensor(users.copy(), dtype=torch.long),
        torch.tensor(pos_movies.copy(), dtype=torch.long),
        torch.tensor(neg_movies.copy(), dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_ndcg = -float("inf")
    best_recall = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for user_batch, pos_batch, neg_batch in loader:
            user_batch = user_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            optimizer.zero_grad()
            pos_scores = model.score(user_batch, pos_batch)
            neg_scores = model.score(user_batch, neg_batch)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()

            batch_n = len(user_batch)
            total_loss += loss.item() * batch_n
            total_samples += batch_n

        mean_loss = total_loss / total_samples
        metrics = evaluate_ranking(model, fit_df, val_pos_df, n_movies, device)

        row = {
            "epoch": epoch,
            "bpr_loss": mean_loss,
            "val_recall@10": metrics["Recall@10"],
            "val_ndcg@10": metrics["NDCG@10"]
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | Loss {mean_loss:.4f} | "
            f"Recall@10 {metrics['Recall@10']:.4f} | "
            f"NDCG@10 {metrics['NDCG@10']:.4f}"
        )

        if metrics["NDCG@10"] > best_ndcg:
            best_ndcg = metrics["NDCG@10"]
            best_recall = metrics["Recall@10"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    # Final fit: retrain on all training interactions for the selected number of epochs.
    print(f"\nTraining final model on all train data for {best_epoch} epochs...")

    final_pos_df = train_model.loc[train_model["rating"] >= 4].copy()

    final_pos_by_user = (final_pos_df.groupby("user_idx")["movie_idx"].apply(set).to_dict())

    final_users, final_pos, final_neg = generate_bpr_pairs(
        final_pos_df,
        final_pos_by_user,
        np.arange(n_movies, dtype=np.int64)
    )

    final_dataset = TensorDataset(
        torch.tensor(final_users.copy(), dtype=torch.long),
        torch.tensor(final_pos.copy(), dtype=torch.long),
        torch.tensor(final_neg.copy(), dtype=torch.long)
    )

    final_loader = DataLoader(final_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MatrixFactorization(n_users=n_users, n_movies=n_movies, emb_dim=EMB_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, best_epoch + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for user_batch, pos_batch, neg_batch in final_loader:
            user_batch = user_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            optimizer.zero_grad()
            pos_scores = model.score(user_batch, pos_batch)
            neg_scores = model.score(user_batch, neg_batch)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()

            batch_n = len(user_batch)
            total_loss += loss.item() * batch_n
            total_samples += batch_n

        print(
            f"Final epoch {epoch}/{best_epoch} | "
            f"Loss: {total_loss / total_samples:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_movies": n_movies,
            "user_ids": data["user_ids"].tolist(),
            "movie_ids": data["movie_ids"].tolist(),
            "embedding_dim": EMB_DIM,
            "best_epoch": best_epoch,
        },
        MODEL_DIR / "bpr_model.pt"
    )

    pd.DataFrame(history).to_csv(REPORTS_DIR / "bpr_training_history.csv", index=False)

    with open(MODEL_DIR / "bpr_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_dim": EMB_DIM,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "best_epoch": best_epoch,
                "val_recall@10": best_recall,
                "val_ndcg@10": best_ndcg,
                "n_users": n_users,
                "n_movies": n_movies,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation Recall@10: {best_recall:.4f}")
    print(f"Best validation NDCG@10: {best_ndcg:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'bpr_model.pt'}")
    print(f"History saved to: {REPORTS_DIR / 'bpr_training_history.csv'}")

if __name__ == "__main__":
    main()