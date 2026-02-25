"""Plots und Tabellen für die HPO-Ergebnisse.

Funktionen:
  save_summary_csv()           — kompakte CSV: bestes AUROC + Accuracy pro Run
  plot_cross_dataset_comparison() — Balkendiagramm über alle Datensätze (AUROC + Accuracy)
  plot_comparison()            — Boxplots pro Datensatz (AUROC + Accuracy)
  plot_convergence()           — Best-so-far AUROC vs. Trial
  plot_training_time()         — Kumulierte Trainingszeit
  plot_efficiency()            — Zeit vs. AUROC Scatter
  plot_energy()                — Energieverbrauch
  plot_hp_distributions()      — HP-Sampling-Histogramme
  plot_search_space_2d()       — Konzeptbild Suchraum (BO vs. Hybrid)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Konzeptbild: Suchraum-Visualisierung (RS / GS / Hybrid)
# ---------------------------------------------------------------------------

def plot_search_space_2d(output_dir):
    """Konzeptbild mit 2 Subplots: Bayesian Optimization (TPE) und Hybrid Search.

    X-Achse: log10(learning_rate), Y-Achse: hidden_dim.
    Keine echten Daten — illustratives Konzeptbild für die Thesis.
    Speichert als search_space_2d.pdf.
    """
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Achsenbeschriftungen
    xlabel = r"$\log_{10}(\mathrm{learning\_rate})$"
    ylabel = "hidden\_dim"
    lr_range = (-5, -1)
    hd_range = (32, 256)

    # --- Subplot 1: Bayesian Optimization (TPE) ---
    ax = axes[0]
    # Warm-up: gleichmäßig gestreute Punkte (hellblau, transparent)
    n_warmup = 10
    lr_warmup = rng.uniform(*lr_range, n_warmup)
    hd_warmup = rng.uniform(*hd_range, n_warmup)
    ax.scatter(lr_warmup, hd_warmup, color="#4C72B0", s=35, alpha=0.4, label="Warm-up (uniform)")

    # TPE-Phase: konzentrierte Samples nahe dem besten Bereich (l(λ)/g(λ))
    n_tpe = 20
    best_lr_bo, best_hd_bo = -2.5, 160.0
    lr_tpe = rng.normal(best_lr_bo, 0.4, n_tpe).clip(*lr_range)
    hd_tpe = rng.normal(best_hd_bo, 25.0, n_tpe).clip(*hd_range)
    ax.scatter(lr_tpe, hd_tpe, color="#4C72B0", s=40, alpha=0.85, label="TPE samples")

    # Bester Fund (Stern)
    ax.scatter([best_lr_bo], [best_hd_bo], color="#1A3A6B", s=160, marker="*", zorder=5,
               label="Best config")

    ax.set_title("Bayesian Optimization (TPE)", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lr_range)
    ax.set_ylim(hd_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- Subplot 2: Hybrid Search (RS + Coordinate-wise Grid) ---
    ax = axes[1]
    # Phase 1: zufällige Punkte (rot)
    n_p1 = 10
    lr_p1 = rng.uniform(*lr_range, n_p1)
    hd_p1 = rng.uniform(*hd_range, n_p1)
    ax.scatter(lr_p1, hd_p1, color="#C44E52", s=40, alpha=0.7, label="Phase 1 (RS)")

    # Bestes aus Phase 1 (Stern)
    best_idx = rng.randint(0, n_p1)
    best_lr, best_hd = lr_p1[best_idx], hd_p1[best_idx]
    ax.scatter([best_lr], [best_hd], color="#C44E52", s=150, marker="*", zorder=5,
               label="Best P1 config")

    # Phase 2: achsenparallele Kreuz-Gitter um bestes Config (orange)
    n_p2 = 5
    lr_p2 = np.linspace(*lr_range, n_p2)
    hd_p2 = np.linspace(*hd_range, n_p2)
    # Horizontale Linie (lr variiert, hd fixiert)
    ax.scatter(lr_p2, [best_hd] * n_p2, color="#FF7F0E", marker="x", s=60, linewidths=1.5,
               label="Phase 2 (Grid)")
    # Vertikale Linie (hd variiert, lr fixiert)
    ax.scatter([best_lr] * n_p2, hd_p2, color="#FF7F0E", marker="x", s=60, linewidths=1.5)

    ax.set_title("Hybrid Search (RS + Grid)", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lr_range)
    ax.set_ylim(hd_range)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Search Space Visualisation (Concept)", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "search_space_2d.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Konvergenz-Plot
# ---------------------------------------------------------------------------

def plot_convergence(df, results_dir):
    """Best-so-far Val-AUROC vs. Trial-Nummer.

    Pro Datensatz + Modell ein Plot, über Seeds gemittelt.
    """
    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        for model_name in df_ds["model"].unique():
            df_model = df_ds[df_ds["model"] == model_name]
            fig, ax = plt.subplots(figsize=(8, 5))

            for optimizer in df_model["optimizer"].unique():
                df_opt = df_model[df_model["optimizer"] == optimizer]

                best_per_seed = []
                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    best_so_far = df_seed["val_auroc"].cummax().values
                    best_per_seed.append(best_so_far)

                min_len = min(len(s) for s in best_per_seed)
                best_per_seed = np.array([s[:min_len] for s in best_per_seed])

                mean = best_per_seed.mean(axis=0)
                std = best_per_seed.std(axis=0)
                trials = np.arange(1, min_len + 1)

                ax.plot(trials, mean, label=optimizer)
                ax.fill_between(trials, mean - std, mean + std, alpha=0.2)

            ax.set_xlabel("Trial")
            ax.set_ylabel("Best Val-AUROC (so far)")
            ax.set_title(f"Konvergenz — {model_name} ({dataset_name})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(results_dir, f"convergence_{dataset_name}_{model_name}.png"),
                dpi=150,
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Vergleichs-Boxplot
# ---------------------------------------------------------------------------

def plot_comparison(df, results_dir):
    """Boxplot: bester AUROC und beste Accuracy pro Seed (BO vs. Hybrid).

    Pro Datensatz: 2 Zeilen (AUROC, Accuracy) × len(models) Spalten.
    """
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    metrics = [
        ("val_auroc",    "Best Val-AUROC"),
        ("val_accuracy", "Best Val-Accuracy"),
    ]

    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        models = sorted(df_ds["model"].unique())
        n_models = len(models)

        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 9))
        # axes shape: (2, n_models) — sicherstellen auch wenn n_models == 1
        axes = np.atleast_2d(axes)

        for row_idx, (metric_col, metric_label) in enumerate(metrics):
            for col_idx, model_name in enumerate(models):
                ax = axes[row_idx, col_idx]
                df_model = df_ds[df_ds["model"] == model_name]

                best = (
                    df_model.groupby(["optimizer", "seed"])[metric_col]
                    .max()
                    .reset_index()
                )
                optimizers = sorted(best["optimizer"].unique())
                data = [
                    best[best["optimizer"] == opt][metric_col].values
                    for opt in optimizers
                ]

                bp = ax.boxplot(data, labels=optimizers, patch_artist=True)
                for patch, color in zip(bp["boxes"], palette[:len(optimizers)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel(metric_label)
                ax.set_title(model_name if row_idx == 0 else "")
                ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Vergleich: AUROC und Accuracy pro Seed ({dataset_name})")
        fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"comparison_boxplot_{dataset_name}.png"),
            dpi=150,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Trainingszeit
# ---------------------------------------------------------------------------

def plot_training_time(df, results_dir):
    """Balkendiagramm: kumulative Trainingszeit pro Methode × Modell × Datensatz."""
    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        cum_time = df_ds.groupby(["model", "optimizer", "seed"])["train_time"].sum().reset_index()
        mean_time = cum_time.groupby(["model", "optimizer"])["train_time"].agg(["mean", "std"]).reset_index()

        models = sorted(mean_time["model"].unique())
        optimizers = sorted(mean_time["optimizer"].unique())
        n_opt = len(optimizers)
        width = 0.8 / n_opt
        x = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, opt in enumerate(optimizers):
            subset = mean_time[mean_time["optimizer"] == opt]
            vals = [subset[subset["model"] == m]["mean"].values[0] for m in models]
            errs = [subset[subset["model"] == m]["std"].values[0] for m in models]
            ax.bar(x + i * width, vals, width, yerr=errs, label=opt, capsize=4)

        ax.set_xticks(x + width * (n_opt - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Kumulative Trainingszeit (s)")
        ax.set_title(f"Trainingszeit ({dataset_name})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"training_time_{dataset_name}.png"),
            dpi=150,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Effizienz-Plot
# ---------------------------------------------------------------------------

def plot_efficiency(df, results_dir):
    """Scatter: kumulierte Trainingszeit (X) vs. bester bisheriger AUROC (Y)."""
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    all_optimizers = sorted(df["optimizer"].unique())
    colors_map = {opt: palette[i % len(palette)] for i, opt in enumerate(all_optimizers)}

    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        for model_name in df_ds["model"].unique():
            df_model = df_ds[df_ds["model"] == model_name]
            fig, ax = plt.subplots(figsize=(8, 5))

            for optimizer in df_model["optimizer"].unique():
                df_opt = df_model[df_model["optimizer"] == optimizer]

                for seed in df_opt["seed"].unique():
                    df_seed = df_opt[df_opt["seed"] == seed].sort_values("trial_nr")
                    cum_time = df_seed["train_time"].cumsum().values
                    best_so_far = df_seed["val_auroc"].cummax().values
                    ax.plot(cum_time, best_so_far, color=colors_map[optimizer],
                            alpha=0.4, linewidth=1)

                ax.plot([], [], color=colors_map[optimizer], label=optimizer)

            ax.set_xlabel("Kumulierte Trainingszeit (s)")
            ax.set_ylabel("Best Val-AUROC (so far)")
            ax.set_title(f"Effizienz — {model_name} ({dataset_name})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(results_dir, f"efficiency_{dataset_name}_{model_name}.png"),
                dpi=150,
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Energie-Plot
# ---------------------------------------------------------------------------

def plot_energy(df, results_dir):
    """Balkendiagramm: Gesamt-Energieverbrauch (kWh) pro Methode × Modell × Datensatz."""
    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        cum_energy = df_ds.groupby(["model", "optimizer", "seed"])["energy_kwh"].sum().reset_index()
        mean_energy = cum_energy.groupby(["model", "optimizer"])["energy_kwh"].agg(["mean", "std"]).reset_index()

        models = sorted(mean_energy["model"].unique())
        optimizers = sorted(mean_energy["optimizer"].unique())
        n_opt = len(optimizers)
        width = 0.8 / n_opt
        x = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, opt in enumerate(optimizers):
            subset = mean_energy[mean_energy["optimizer"] == opt]
            vals = [subset[subset["model"] == m]["mean"].values[0] for m in models]
            errs = [subset[subset["model"] == m]["std"].values[0] for m in models]
            ax.bar(x + i * width, vals, width, yerr=errs, label=opt, capsize=4)

        ax.set_xticks(x + width * (n_opt - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Energieverbrauch (kWh)")
        ax.set_title(f"Energieverbrauch ({dataset_name})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"energy_{dataset_name}.png"),
            dpi=150,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# HP-Verteilungen
# ---------------------------------------------------------------------------

def plot_hp_distributions(df, results_dir):
    """Histogramme der gewählten HP-Werte pro Datensatz × Modell."""
    hp_cols = [c for c in df.columns if c.startswith("hp_")]
    if not hp_cols:
        return

    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name]
        for model_name in df_ds["model"].unique():
            df_model = df_ds[df_ds["model"] == model_name]

            valid_hps = [c for c in hp_cols if df_model[c].notna().any()]
            if not valid_hps:
                continue

            n_hps = len(valid_hps)
            n_cols = min(3, n_hps)
            n_rows = (n_hps + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_hps == 1:
                axes = np.array([axes])
            axes = np.atleast_2d(axes)

            optimizers = sorted(df_model["optimizer"].unique())

            for idx, hp_col in enumerate(valid_hps):
                row, col = divmod(idx, n_cols)
                ax = axes[row, col]

                for opt in optimizers:
                    values = df_model[df_model["optimizer"] == opt][hp_col].dropna()
                    ax.hist(values, bins=20, alpha=0.5, label=opt)

                ax.set_title(hp_col.replace("hp_", ""))
                ax.set_xlabel("Wert")
                ax.set_ylabel("Häufigkeit")
                ax.legend()

            for idx in range(n_hps, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row, col].set_visible(False)

            fig.suptitle(f"HP-Verteilungen — {model_name} ({dataset_name})")
            fig.tight_layout()
            fig.savefig(
                os.path.join(results_dir, f"hp_distributions_{dataset_name}_{model_name}.png"),
                dpi=150,
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Zusammenfassende CSV: bestes AUROC + Accuracy pro Run
# ---------------------------------------------------------------------------

def _trials_to_pct_best(series_sorted, pct=0.95):
    """Gibt den 1-basierten Trial-Index zurück, ab dem cummax >= pct * max erreicht wird.

    Args:
        series_sorted: val_auroc-Werte eines Runs, nach trial_nr sortiert.
        pct: Schwellwert-Anteil des besten AUROC (Standard: 0.95).
    Returns:
        int: erster Trial-Index (1-basiert) der den Schwellwert überschreitet.
             Gibt len(series) zurück, falls der Schwellwert nie erreicht wird.
    """
    best = series_sorted.max()
    threshold = pct * best
    cummax = series_sorted.cummax()
    reached = cummax >= threshold
    if not reached.any():
        return len(series_sorted)
    return int(reached.idxmax()) + 1  # idxmax gibt 0-basierten Positionsindex


def save_summary_csv(df, results_dir):
    """Speichert eine kompakte Tabelle: ein Eintrag pro Run (dataset × model × optimizer × seed).

    Spalten: dataset, model, optimizer, seed,
             best_auroc, best_accuracy,
             trials_to_95pct_auroc,   ← erster Trial mit AUROC >= 95 % des Run-Maximums
             total_train_time_s, total_energy_kwh, n_trials
    Ausgabe: results/summary_best_per_run.csv
    """
    rows = []
    for (dataset, model, optimizer, seed), grp in df.groupby(
        ["dataset", "model", "optimizer", "seed"]
    ):
        grp_sorted = grp.sort_values("trial_nr").reset_index(drop=True)
        auroc_series = grp_sorted["val_auroc"].reset_index(drop=True)

        rows.append({
            "dataset":             dataset,
            "model":               model,
            "optimizer":           optimizer,
            "seed":                seed,
            "best_auroc":          auroc_series.max(),
            "best_accuracy":       grp_sorted["val_accuracy"].max(),
            "trials_to_95pct_auroc": _trials_to_pct_best(auroc_series, pct=0.95),
            "total_train_time_s":  grp_sorted["train_time"].sum(),
            "total_energy_kwh":    grp_sorted["energy_kwh"].sum(),
            "n_trials":            len(grp_sorted),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model", "optimizer", "seed"])
        .reset_index(drop=True)
    )
    out_path = os.path.join(results_dir, "summary_best_per_run.csv")
    summary.to_csv(out_path, index=False)
    print(f"[plots] Saved: {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Cross-Dataset-Vergleich: AUROC + Accuracy über alle Datensätze
# ---------------------------------------------------------------------------

def plot_cross_dataset_comparison(df, results_dir):
    """Balkendiagramm: mittlerer bester AUROC und Accuracy je Datensatz × Optimizer.

    Pro Modell (MLP, RF) eine Figure mit 2 Zeilen:
      Zeile 1 — Mean Best Val-AUROC (Fehlerbalken = std über Seeds)
      Zeile 2 — Mean Best Val-Accuracy (Fehlerbalken = std über Seeds)
    X-Achse: Datensätze; Balkengruppen: BO vs. Hybrid.
    Ausgabe: results/cross_dataset_{model}.png
    """
    palette = {"bo": "#4C72B0", "hybrid": "#DD8452"}
    metrics = [
        ("val_auroc",    "Mean Best Val-AUROC"),
        ("val_accuracy", "Mean Best Val-Accuracy"),
    ]
    datasets   = sorted(df["dataset"].unique())
    optimizers = sorted(df["optimizer"].unique())
    x = np.arange(len(datasets))
    width = 0.35

    for model_name in sorted(df["model"].unique()):
        df_model = df[df["model"] == model_name]
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        for row_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx]

            for opt_idx, opt in enumerate(optimizers):
                df_opt = df_model[df_model["optimizer"] == opt]
                means, stds = [], []
                for ds in datasets:
                    best_per_seed = (
                        df_opt[df_opt["dataset"] == ds]
                        .groupby("seed")[metric_col]
                        .max()
                    )
                    means.append(best_per_seed.mean())
                    stds.append(best_per_seed.std())

                offset = (opt_idx - (len(optimizers) - 1) / 2) * width
                ax.bar(
                    x + offset, means, width, yerr=stds,
                    label=opt.upper(),
                    color=palette.get(opt, "#888"),
                    alpha=0.85, capsize=5,
                )

            ax.set_ylabel(metric_label)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(datasets, rotation=15, ha="right")
        fig.suptitle(f"Cross-Dataset Comparison — {model_name} (mean ± std over seeds)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"cross_dataset_{model_name}.png"),
            dpi=150,
        )
        plt.close(fig)
        print(f"[plots] Saved: cross_dataset_{model_name}.png")
