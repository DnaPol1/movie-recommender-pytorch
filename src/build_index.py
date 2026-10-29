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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model = MatrixFactorization(
        n_users=checkpoint["n_users"],
        n_movies=checkpoint["n_movies"],
        emb_dim=checkpoint["embedding_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Extract movie embeddings and biases.
    with torch.no_grad():
        movie_vectors = model.movie_emb.weight.detach().cpu().numpy()
        movie_biases = model.movie_bias.weight.detach().cpu().numpy()

    movie_vectors = movie_vectors.astype(np.float32)
    movie_biases = movie_biases.astype(np.float32)

    # Add movie bias as an extra coordinate.
    # Query vectors will have an extra coordinate equal to 1.
    index_vectors = np.concatenate([movie_vectors, movie_biases], axis=1).copy()

    # Exact maximum-inner-product search.
    index = faiss.IndexFlatIP(index_vectors.shape[1])
    index.add(index_vectors)

    # Keep the mapping from FAISS row indices to original movie IDs.
    movie_ids = np.asarray(checkpoint["movie_ids"], dtype=np.int64)

    if len(movie_ids) != index.ntotal:
        raise ValueError("Movie ID count does not match the number of indexed vectors.")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(MOVIE_IDS_PATH, movie_ids)

    print(f"Indexed movies: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Movie ID mapping saved to: {MOVIE_IDS_PATH}")


if __name__ == "__main__":
    main()