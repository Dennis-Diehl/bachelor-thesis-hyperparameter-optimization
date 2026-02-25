"""Modelle: MLP (PyTorch) und Random-Forest-Wrapper (sklearn)."""

import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class MLP(nn.Module):
    """Flexibles MLP mit konfigurierbarer Anzahl Hidden Layers und Aktivierungsfunktion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        output_dim: int = 2,
        num_layers: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RFModel:
    """Wrapper um sklearn RandomForestClassifier mit einheitlichem Interface."""

    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
