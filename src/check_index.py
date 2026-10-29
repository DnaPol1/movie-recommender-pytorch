from pathlib import Path

import faiss
import numpy as np
import torch

from src.model import MatrixFactorization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bpr_model.pt"
INDEX_PATH = PROJECT_ROOT / "models" / "movie_index.faiss"
MOVIE_IDS_PATH = PROJECT_ROOT / "models" / "movie_index_ids.npy"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model = MatrixFactorization(
        n_users=checkpoint["n_users"],
        n_movies=checkpoint["n_movies"],
        emb_dim=checkpoint["embedding_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    index = faiss.read_index(str(INDEX_PATH))
    movie_ids = np.load(MOVIE_IDS_PATH)

    # Pick one user to test.
    user_idx = 0
    top_k = 10

    with torch.no_grad():
        user_vector = model.user_emb.weight[user_idx].cpu().numpy()
        query = np.append(user_vector, 1.0).astype(np.float32)

        _, faiss_positions = index.search(query.reshape(1, -1), top_k)

        faiss_movie_ids = movie_ids[faiss_positions[0]]

        # Calculate scores directly with the model.
        all_movie_idx = torch.arange(checkpoint["n_movies"], dtype=torch.long, device=device)
        user_tensor = torch.full_like(all_movie_idx, user_idx)

        model_scores = model.score(user_tensor, all_movie_idx)
        direct_positions = torch.topk(model_scores, top_k).indices.cpu().numpy()
        direct_movie_ids = movie_ids[direct_positions]

    print("FAISS top-10 movie IDs:")
    print(faiss_movie_ids.tolist())

    print("\nDirect model top-10 movie IDs:")
    print(direct_movie_ids.tolist())

    same_order = np.array_equal(faiss_movie_ids, direct_movie_ids)
    same_set = set(faiss_movie_ids) == set(direct_movie_ids)

    print(f"\nSame order: {same_order}")
    print(f"Same top-10 set: {same_set}")

if __name__ == "__main__":
    main()