"""Training, Evaluation und Ressourcenmessung für MLP und Random Forest."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from codecarbon import EmissionsTracker

from models import MLP, RFModel
from config import MAX_EPOCHS, EARLY_STOPPING_PATIENCE


def _train_mlp_loop(
    model,
    train_loader,
    val_loader,
    lr,
    epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    optimizer_name="adam",
    weight_decay=0.0,
):
    """Trainiert MLP mit Early Stopping auf Val-Loss. Gibt Modell + History zurück."""
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # adam (default)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history


def evaluate_mlp(model, loader):
    """Berechnet Accuracy und AUROC für ein MLP auf einem DataLoader."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    return acc, auroc


def _train_rf_model(params, X_train, y_train):
    """Trainiert Random Forest und gibt das Modell zurück."""
    model = RFModel(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_rf(model, X, y):
    """Berechnet Accuracy und AUROC für Random Forest."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auroc = roc_auc_score(y, probs)
    return acc, auroc


def measure_resources_mlp(train_fn, eval_fn):
    """Misst Trainingszeit, Inferenzzeit und Energie für MLP.

    Args:
        train_fn: Callable das Training durchführt und (trained_model, history) zurückgibt.
        eval_fn: Callable das Evaluation durchführt.

    Returns:
        dict mit train_time, inference_time, energy_kwh
    """
    # Trainingszeit + Energie
    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    t0 = time.perf_counter()
    trained_model, history = train_fn()
    train_time = time.perf_counter() - t0
    emissions = tracker.stop()

    # Inferenzzeit
    t0 = time.perf_counter()
    acc, auroc = eval_fn(trained_model)
    inference_time = time.perf_counter() - t0

    return {
        "trained_model": trained_model,
        "history": history,
        "val_accuracy": acc,
        "val_auroc": auroc,
        "train_time": train_time,
        "inference_time": inference_time,
        "energy_kwh": emissions if emissions is not None else 0.0,
    }


def measure_resources_rf(train_fn, eval_fn, X_val):
    """Misst Trainingszeit, Inferenzzeit und Energie für Random Forest.

    Args:
        train_fn: Callable das Training durchführt und das Modell zurückgibt.
        eval_fn: Callable das Evaluation durchführt.
        X_val: Gesamtes Validierungs-Set für Inferenzzeit-Messung.

    Returns:
        dict mit train_time, inference_time, energy_kwh
    """
    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    t0 = time.perf_counter()
    trained_model = train_fn()
    train_time = time.perf_counter() - t0
    emissions = tracker.stop()

    # Evaluation auf Val-Set
    acc, auroc = eval_fn(trained_model)

    # Inferenzzeit (gesamtes Val-Set)
    t0 = time.perf_counter()
    trained_model.predict(X_val)
    inference_time = time.perf_counter() - t0

    return {
        "trained_model": trained_model,
        "val_accuracy": acc,
        "val_auroc": auroc,
        "train_time": train_time,
        "inference_time": inference_time,
        "energy_kwh": emissions if emissions is not None else 0.0,
    }


def train_mlp(X_train, y_train, X_val, y_val, config) -> dict:
    """Wrapper: nimmt numpy Arrays, gibt Metriken-Dict zurück.

    Intern: DataLoader erstellen, Modell bauen, trainieren, evaluieren.
    """
    from data import to_dataloaders
    loaders = to_dataloaders(
        {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val},
        batch_size=config["batch_size"],
    )
    input_dim = X_train.shape[1]
    model = MLP(
        input_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_layers=config.get("num_layers", 1),
        activation=config.get("activation", "relu"),
    )
    return measure_resources_mlp(
        train_fn=lambda: _train_mlp_loop(
            model,
            loaders["train"],
            loaders["val"],
            lr=config["learning_rate"],
            optimizer_name=config.get("optimizer_name", "adam"),
            weight_decay=config.get("weight_decay", 0.0),
        ),
        eval_fn=lambda m: evaluate_mlp(m, loaders["val"]),
    )


def train_rf(X_train, y_train, X_val, y_val, config, seed=42) -> dict:
    """Wrapper: nimmt numpy Arrays, gibt Metriken-Dict zurück.

    Intern: Modell bauen, trainieren, evaluieren.
    """
    params = {**config, "random_state": seed}
    return measure_resources_rf(
        train_fn=lambda: _train_rf_model(params, X_train, y_train),
        eval_fn=lambda m: evaluate_rf(m, X_val, y_val),
        X_val=X_val,
    )
