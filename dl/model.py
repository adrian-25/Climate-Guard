"""dl/model.py — GRU/LSTM sequence classifier with city embedding.

Architecture
------------
Input  : (batch, seq_len, n_features)  float32
         + city_idx (batch,)           int64

 ┌─────────────────────────────────────────────┐
 │  Learned city embedding  (num_cities → 4)   │
 │  GRU or LSTM  (n_features → hidden, 1-2L)   │
 │     take last hidden state  (hidden,)        │
 │  concat(last_hidden, city_embed)             │
 │  MLP head: Linear→ReLU→Dropout→Linear(1)    │
 │  single logit → sigmoid in inference         │
 └─────────────────────────────────────────────┘

Output : raw logit (batch, 1)  for use with BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    CITY_EMBED_DIM,
    DROPOUT,
    GRU_HIDDEN,
    LSTM_HIDDEN,
    MLP_HIDDEN,
    NUM_INDIA_CITIES,
    NUM_LAYERS,
)


class HeatwaveRNN(nn.Module):
    """
    GRU or LSTM-based sequence classifier for heatwave risk prediction.

    Parameters
    ----------
    n_features    : input feature dimension
    arch          : "gru" or "lstm"
    num_cities    : number of distinct cities (for embedding)
    hidden        : RNN hidden size
    num_layers    : RNN layers
    dropout       : dropout probability
    embed_dim     : city embedding dimension
    mlp_hidden    : hidden units in the MLP head
    """

    def __init__(
        self,
        n_features: int,
        arch: str = "gru",
        num_cities: int = NUM_INDIA_CITIES,
        hidden: int = GRU_HIDDEN,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        embed_dim: int = CITY_EMBED_DIM,
        mlp_hidden: int = MLP_HIDDEN,
    ) -> None:
        super().__init__()
        arch = arch.lower()
        assert arch in {"gru", "lstm"}, f"arch must be 'gru' or 'lstm', got {arch!r}"
        self.arch = arch

        self.city_embed = nn.Embedding(num_cities, embed_dim)

        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        head_in = hidden + embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, n_features)
        city: torch.Tensor,  # (batch,)  int64
    ) -> torch.Tensor:  # (batch, 1)  raw logit
        # RNN
        if self.arch == "gru":
            out, h = self.rnn(x)  # h: (layers, batch, hidden)
            last = h[-1]  # (batch, hidden)
        else:
            out, (h, _) = self.rnn(x)
            last = h[-1]

        # City embedding
        emb = self.city_embed(city)  # (batch, embed_dim)

        # Concatenate and classify
        z = torch.cat([last, emb], dim=1)  # (batch, hidden + embed_dim)
        return self.head(z)  # (batch, 1)


def build_model(
    n_features: int,
    arch: str = "gru",
    num_cities: int = NUM_INDIA_CITIES,
    **kwargs,
) -> HeatwaveRNN:
    """Convenience factory; pass extra kwargs to override defaults."""
    return HeatwaveRNN(n_features=n_features, arch=arch, num_cities=num_cities, **kwargs)
