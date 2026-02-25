"""Entry Point für den HPO-Benchmark.

Hier läuft alles zusammen: Daten laden, Experimente durchführen
(2 Modelle × 2 Optimierer × 3 Seeds × 3 Datensätze = 36 Runs), Ergebnisse als CSV speichern
und alle Plots erzeugen.
"""

import json
import os
import platform
import random

import numpy as np
import pandas as pd
import ray
import torch

from config import N_TRIALS, SEEDS, DATASETS
from data import load_data
from optimize import bayesian_search, hybrid_search
from plots import (
    plot_comparison,
    plot_convergence,
    plot_cross_dataset_comparison,
    plot_efficiency,
    plot_energy,
    plot_hp_distributions,
    plot_training_time,
    plot_search_space_2d,
    save_summary_csv,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def set_seed(seed):
    """Alle relevanten Seeds setzen, damit Experimente reproduzierbar sind."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hardware_info():
    """Hardware- und Software-Infos sammeln — für die Thesis-Dokumentation."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Hardware-Info ausgeben und für später speichern
    hw_info = get_hardware_info()
    print("Hardware Info:")
    for k, v in hw_info.items():
        print(f"  {k}: {v}")
    with open(os.path.join(RESULTS_DIR, "hardware_info.json"), "w") as f:
        json.dump(hw_info, f, indent=2)

    # Konzept-Visualisierung des Suchraums (einmalig, ohne echte Daten)
    gfx_dir = os.path.join(os.path.dirname(__file__), "Thesis-Ausarbeitung", "gfx")
    os.makedirs(gfx_dir, exist_ok=True)
    plot_search_space_2d(gfx_dir)

    # Ray einmal starten — alle tune.run()-Aufrufe nutzen dieselbe Instanz
    ray.init(ignore_reinit_error=True)

    # 4 Kombinationen pro Seed × Datensatz
    # (model_type, Anzeigename, Such-Funktion, Kurzname für CSV)
    experimente = [
        ("mlp", "MLP", bayesian_search, "bo"),
        ("mlp", "MLP", hybrid_search,   "hybrid"),
        ("rf",  "RF",  bayesian_search, "bo"),
        ("rf",  "RF",  hybrid_search,   "hybrid"),
    ]

    alle_zeilen = []

    for dataset_name in DATASETS:
        print(f"\n{'#'*70}")
        print(f"  DATENSATZ: {dataset_name}")
        print(f"{'#'*70}")

        for seed in SEEDS:
            set_seed(seed)
            data_np, _ = load_data(seed, dataset=dataset_name)

            for model_type, model_name, search_fn, optimizer_name in experimente:
                print(f"\n{'='*60}")
                print(f"  Dataset={dataset_name}  Modell={model_name}  "
                      f"Optimizer={optimizer_name.upper()}  Seed={seed}")
                print(f"{'='*60}")

                ergebnisse = search_fn(model_type, seed, N_TRIALS, data_np)

                for trial_nr, zeile in enumerate(ergebnisse, start=1):
                    zeile.update({
                        "dataset":   dataset_name,
                        "model":     model_name,
                        "optimizer": optimizer_name,
                        "seed":      seed,
                        "trial_nr":  trial_nr,
                    })
                    alle_zeilen.append(zeile)

    ray.shutdown()

    # Alles in einen DataFrame packen und als CSV rausschreiben
    df = pd.DataFrame(alle_zeilen)
    csv_pfad = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(csv_pfad, index=False)
    print(f"\nErgebnisse gespeichert: {csv_pfad}")

    # Kurze Zusammenfassung im Terminal
    print("\nVal-AUROC pro Datensatz, Modell und Optimierer (mean ± std über die Seeds):")
    print(
        df.groupby(["dataset", "model", "optimizer"])["val_auroc"]
        .agg(["mean", "std", "max"])
        .round(4)
    )
    print("\nVal-Accuracy pro Datensatz, Modell und Optimierer (mean ± std über die Seeds):")
    print(
        df.groupby(["dataset", "model", "optimizer"])["val_accuracy"]
        .agg(["mean", "std", "max"])
        .round(4)
    )

    # Zusammenfassende CSV (ein Eintrag pro Run: bestes AUROC + Accuracy)
    save_summary_csv(df, RESULTS_DIR)

    # Alle Plots erzeugen
    print("\nErzeuge Plots...")
    plot_convergence(df, RESULTS_DIR)
    plot_comparison(df, RESULTS_DIR)
    plot_cross_dataset_comparison(df, RESULTS_DIR)
    plot_training_time(df, RESULTS_DIR)
    plot_efficiency(df, RESULTS_DIR)
    plot_energy(df, RESULTS_DIR)
    plot_hp_distributions(df, RESULTS_DIR)
    print(f"Plots gespeichert in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
