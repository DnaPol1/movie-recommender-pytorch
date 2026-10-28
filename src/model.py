import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """Matrix Factorization model trained with the BPR objective."""

    def __init__(self, n_users, n_movies, emb_dim=16):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        self.global_mean = nn.Parameter(torch.tensor(3.5))

        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.movie_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def score(self, user_idx, movie_idx):
        user_vec = self.user_emb(user_idx)
        movie_vec = self.movie_emb(movie_idx)

        interaction = (user_vec * movie_vec).sum(dim=1)

        return (
            self.global_mean
            + self.user_bias(user_idx).squeeze(-1)
            + self.movie_bias(movie_idx).squeeze(-1)
            + interaction
        )

    def forward(self, user_idx, movie_idx):
        return self.score(user_idx, movie_idx)