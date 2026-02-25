"""Hyperparameter-Optimierung mit Ray Tune für MLP und Random Forest.

Hier laufen die zwei Strategien:
  - bayesian_search(): TPE via OptunaSearch — lernt aus vorherigen Trials
  - hybrid_search():   Phase 1 Random Search → Phase 2 per-HP Grid Search
                       (für jede Iteration k wiederholt)

Alle Funktionen geben eine Liste von Trial-Ergebnissen zurück, die main.py
dann zu einem DataFrame zusammenbaut und als CSV speichert.
"""

import copy
import math

import numpy as np
from ray import tune
from ray.tune import with_parameters
from ray.tune.search.optuna import OptunaSearch

from config import MLP_SEARCH_SPACE, RF_SEARCH_SPACE
from data import make_loaders
from models import MLP
from train import (
    train_mlp, evaluate_mlp,
    train_rf, evaluate_rf,
    measure_resources_mlp, measure_resources_rf,
)


# ---------------------------------------------------------------------------
# Trial-Funktionen — eine pro Modelltyp
# ---------------------------------------------------------------------------

def _mlp_trial(config, data_np, seed):
    """Einen MLP-Trial durchführen: Modell bauen, trainieren, Metriken reporten."""
    input_dim = data_np["X_train"].shape[1]

    loaders = make_loaders(data_np, batch_size=config["batch_size"])

    model = MLP(
        input_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_layers=config.get("num_layers", 1),
        activation=config.get("activation", "relu"),
    )

    res = measure_resources_mlp(
        train_fn=lambda: train_mlp(
            model,
            loaders["train"],
            loaders["val"],
            lr=config["learning_rate"],
            optimizer_name=config.get("optimizer_name", "adam"),
            weight_decay=config.get("weight_decay", 0.0),
        ),
        eval_fn=lambda m: evaluate_mlp(m, loaders["val"]),
    )

    tune.report({
        "val_auroc":      res["val_auroc"],
        "val_accuracy":   res["val_accuracy"],
        "train_time":     res["train_time"],
        "inference_time": res["inference_time"],
        "energy_kwh":     res["energy_kwh"],
    })


def _rf_trial(config, data_np, seed):
    """Einen RF-Trial durchführen: Modell bauen, trainieren, Metriken reporten."""
    params = {**config, "random_state": seed}

    res = measure_resources_rf(
        train_fn=lambda: train_rf(params, data_np["X_train"], data_np["y_train"]),
        eval_fn=lambda m: evaluate_rf(m, data_np["X_val"], data_np["y_val"]),
        X_val=data_np["X_val"],
    )

    tune.report({
        "val_auroc":      res["val_auroc"],
        "val_accuracy":   res["val_accuracy"],
        "train_time":     res["train_time"],
        "inference_time": res["inference_time"],
        "energy_kwh":     res["energy_kwh"],
    })


# ---------------------------------------------------------------------------
# Hilfsfunktion: Ergebnisse aus dem analysis-Objekt rausziehen
# ---------------------------------------------------------------------------

def _extract_results(analysis, param_keys):
    """Alle Trial-Ergebnisse als Liste von Dicts zurückgeben."""
    rows = []
    for trial in analysis.trials:
        result = trial.last_result
        if not result:
            continue
        row = {
            k: result.get(k)
            for k in ("val_auroc", "val_accuracy", "train_time", "inference_time", "energy_kwh")
        }
        for k in param_keys:
            row[f"hp_{k}"] = trial.config.get(k)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Hilfsfunktion: einzelnen Config-Dict evaluieren (für Hybrid Search)
# ---------------------------------------------------------------------------

def _eval_config(config, model_type, data_np, seed):
    """Evaluiert eine konkrete Konfiguration und gibt val_auroc + Metriken zurück."""
    if model_type == "mlp":
        input_dim = data_np["X_train"].shape[1]
        loaders = make_loaders(data_np, batch_size=config["batch_size"])
        model = MLP(
            input_dim,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            num_layers=config.get("num_layers", 1),
            activation=config.get("activation", "relu"),
        )
        res = measure_resources_mlp(
            train_fn=lambda: train_mlp(
                model,
                loaders["train"],
                loaders["val"],
                lr=config["learning_rate"],
                optimizer_name=config.get("optimizer_name", "adam"),
                weight_decay=config.get("weight_decay", 0.0),
            ),
            eval_fn=lambda m: evaluate_mlp(m, loaders["val"]),
        )
    else:
        params = {**config, "random_state": seed}
        res = measure_resources_rf(
            train_fn=lambda: train_rf(params, data_np["X_train"], data_np["y_train"]),
            eval_fn=lambda m: evaluate_rf(m, data_np["X_val"], data_np["y_val"]),
            X_val=data_np["X_val"],
        )
    return res


def _sample_hp_grid(hp_name, param_space, n_grid):
    """Erzeugt n_grid Kandidaten für einen HP basierend auf dem Ray-Tune-Suchraum."""
    spec = param_space[hp_name]
    # Ray Tune spec-Objekte haben unterschiedliche interne Strukturen.
    # Wir nutzen sample() um den Typ zu erkennen und dann ein Grid zu konstruieren.
    sample_val = spec.sample()

    candidates = []
    if isinstance(sample_val, float):
        # Kontinuierlich: linear oder log
        # Mehrfach samplen und Bereich schätzen
        samples = [spec.sample() for _ in range(50)]
        lo, hi = min(samples), max(samples)
        # Log-Grid falls Verhältnis > 100 (Hinweis auf log-uniform)
        if hi / max(lo, 1e-12) > 100:
            candidates = list(np.logspace(math.log10(lo), math.log10(hi), n_grid))
        else:
            candidates = list(np.linspace(lo, hi, n_grid))
    elif isinstance(sample_val, int):
        samples = [spec.sample() for _ in range(50)]
        lo, hi = min(samples), max(samples)
        candidates = [int(v) for v in np.linspace(lo, hi, n_grid)]
        candidates = sorted(set(candidates))
    elif isinstance(sample_val, str):
        # Categorical: alle Optionen zurückgeben (ignoriert n_grid)
        seen = set()
        for _ in range(200):
            v = spec.sample()
            seen.add(v)
        candidates = sorted(seen)
    else:
        # Unbekannter Typ: n_grid zufällige Samples
        candidates = [spec.sample() for _ in range(n_grid)]

    return candidates


# ---------------------------------------------------------------------------
# Bayesian Optimization (TPE)
# ---------------------------------------------------------------------------

def bayesian_search(model_type: str, seed: int, n_trials: int, data_np: dict) -> list:
    """Bayesian Optimization mit OptunaSearch (TPE) via Ray Tune."""
    if model_type == "mlp":
        trial_fn, param_space = _mlp_trial, MLP_SEARCH_SPACE
    else:
        trial_fn, param_space = _rf_trial, RF_SEARCH_SPACE

    trainable = with_parameters(trial_fn, data_np=data_np, seed=seed)
    search_alg = OptunaSearch(metric="val_auroc", mode="max", seed=seed)

    analysis = tune.run(
        trainable,
        config=param_space,
        search_alg=search_alg,
        num_samples=n_trials,
        metric="val_auroc",
        mode="max",
        max_concurrent_trials=1,
        verbose=1,
    )
    return _extract_results(analysis, list(param_space.keys()))


# ---------------------------------------------------------------------------
# Hybrid Search (RS + Grid Search per HP)
# ---------------------------------------------------------------------------

def hybrid_search(
    model_type: str,
    seed: int,
    n_trials: int,
    data_np: dict,
    n_random: int = 20,
    n_grid: int = 5,
    k_iterations: int = 1,
) -> list:
    """Hybrid-Suche: Phase 1 Random Search → Phase 2 per-HP Grid Search.

    Budget: n_random + n_params * n_grid (pro Iteration k).
    Beispiel (k=1, MLP): 20 + 8*5 = 60 → wir setzen n_random so, dass Budget ≤ n_trials.

    Args:
        model_type:   "mlp" oder "rf"
        seed:         Random State für Reproduzierbarkeit
        n_trials:     Gesamtbudget (Anzahl Evaluierungen)
        data_np:      Daten-Dict
        n_random:     Trials in Phase 1 (Random Search)
        n_grid:       Grid-Punkte pro HP in Phase 2
        k_iterations: Wie oft Phase 1 + Phase 2 wiederholt wird

    Returns:
        Liste von Trial-Ergebnis-Dicts (Metriken + hp_*-Werte)
    """
    if model_type == "mlp":
        trial_fn, param_space = _mlp_trial, MLP_SEARCH_SPACE
    else:
        trial_fn, param_space = _rf_trial, RF_SEARCH_SPACE

    param_keys = list(param_space.keys())
    n_params = len(param_keys)

    # Budget anpassen, falls zu wenig Trials vorhanden
    budget_per_iter = n_random + n_params * n_grid
    if budget_per_iter * k_iterations > n_trials:
        # n_random proportional kürzen
        available = n_trials // k_iterations
        n_random = max(1, available - n_params * n_grid)

    all_rows = []
    trial_counter = 0

    # Startpunkt: zufällige Konfiguration samplen
    best_config = {k: param_space[k].sample() for k in param_keys}
    best_auroc = -1.0

    for k in range(k_iterations):
        print(f"\n[hybrid_search] Iteration {k+1}/{k_iterations} — Phase 1: {n_random} Random Trials")

        # --- Phase 1: Random Search ---
        trainable = with_parameters(trial_fn, data_np=data_np, seed=seed)
        analysis = tune.run(
            trainable,
            config=param_space,
            num_samples=n_random,
            metric="val_auroc",
            mode="max",
            max_concurrent_trials=1,
            verbose=0,
        )
        phase1_rows = _extract_results(analysis, param_keys)
        for row in phase1_rows:
            row["trial_nr_global"] = trial_counter
            trial_counter += 1
        all_rows.extend(phase1_rows)

        # Bestes Config aus Phase 1 ermitteln
        best_phase1 = max(phase1_rows, key=lambda r: r.get("val_auroc") or -1)
        if (best_phase1.get("val_auroc") or -1) > best_auroc:
            best_auroc = best_phase1["val_auroc"]
            best_config = {k: best_phase1[f"hp_{k}"] for k in param_keys if f"hp_{k}" in best_phase1}

        print(f"[hybrid_search] Bestes Config nach Phase 1: auroc={best_auroc:.4f}")

        # --- Phase 2: Per-HP Grid Search ---
        print(f"[hybrid_search] Phase 2: Grid Search über {n_params} HPs × {n_grid} Punkte")
        current_config = copy.deepcopy(best_config)

        for hp_name in param_keys:
            candidates = _sample_hp_grid(hp_name, param_space, n_grid)
            best_val_for_hp = -1.0
            best_candidate = current_config.get(hp_name)

            for cand in candidates:
                test_config = {**current_config, hp_name: cand}
                res = _eval_config(test_config, model_type, data_np, seed)

                row = {
                    "val_auroc":      res["val_auroc"],
                    "val_accuracy":   res["val_accuracy"],
                    "train_time":     res["train_time"],
                    "inference_time": res["inference_time"],
                    "energy_kwh":     res["energy_kwh"],
                    "trial_nr_global": trial_counter,
                }
                for k2 in param_keys:
                    row[f"hp_{k2}"] = test_config.get(k2)
                all_rows.append(row)
                trial_counter += 1

                if res["val_auroc"] > best_val_for_hp:
                    best_val_for_hp = res["val_auroc"]
                    best_candidate = cand

            # Besten Wert für diesen HP übernehmen
            current_config[hp_name] = best_candidate
            if best_val_for_hp > best_auroc:
                best_auroc = best_val_for_hp

        best_config = current_config
        print(f"[hybrid_search] Bestes Config nach Phase 2: auroc={best_auroc:.4f}")

    return all_rows
