"""Datensätze laden, splitten und skalieren."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(seed: int, batch_size: int = 32, dataset: str = "breast_cancer"):
    """Lädt einen Datensatz, splittet 60/20/20, skaliert, gibt numpy + DataLoader zurück.

    Args:
        seed:       Random state für reproduzierbare Splits.
        batch_size: Batch-Größe für die DataLoader.
        dataset:    "breast_cancer" | "diabetes" | "bank_marketing"
    """
    if dataset == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)

    elif dataset == "diabetes":
        # Pima Indians Diabetes (768 Samples, 8 Features, binary)
        data = fetch_openml(name="diabetes", version=1, as_frame=True)
        X = data.data.values.astype(np.float32)
        # Labels: "tested_positive" → 1, sonst → 0
        y = (data.target == "tested_positive").astype(int).values

    elif dataset == "bank_marketing":
        # Bank Marketing (4521 Samples, 16 Features nach Encoding, binary)
        data = fetch_openml(name="bank-marketing", version=1, as_frame=True)
        df = data.data.copy()
        # One-hot-Encoding für kategorische Spalten
        df = pd.get_dummies(df)
        X = df.values.astype(np.float32)
        # Labels: "2" = yes → 1, "1" = no → 0
        y = (data.target == "2").astype(int).values

        # 10 %-Subsample für überschaubare Laufzeit
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=int(0.1 * len(X)), replace=False)
        X, y = X[idx], y[idx]

    else:
        raise ValueError(f"Unbekannter Datensatz: {dataset!r}")

    print(f"[data] {dataset}: X={X.shape}, y={y.shape}, positives={y.sum()}/{len(y)}")

    # Erst Train+Val vs Test (80/20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Dann Train vs Val (75/25 von 80% → 60/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
    )

    # Skalierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Numpy-Daten
    data_np = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    # PyTorch DataLoader
    loaders = make_loaders(data_np, batch_size)

    return data_np, loaders


def make_loaders(data_np, batch_size: int):
    """Erstellt neue DataLoader mit anderem batch_size aus vorhandenen numpy-Daten."""
    def make_loader(X, y, bs, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    return {
        "train": make_loader(data_np["X_train"], data_np["y_train"], batch_size, shuffle=True),
        "val":   make_loader(data_np["X_val"],   data_np["y_val"],   batch_size),
        "test":  make_loader(data_np["X_test"],  data_np["y_test"],  batch_size),
    }
