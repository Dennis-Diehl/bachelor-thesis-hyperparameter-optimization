# Adaptive Coordinate Random Search for Hyperparameter Optimization

Bachelor thesis, Data Mining group, Johannes Gutenberg University Mainz.

## Abstract

Hyperparameter optimization (HPO) is essential for the predictive performance of machine learning models, but the optimization process itself consumes computational resources and energy, an effect Bayesian Optimization (BO) via the Tree-Structured Parzen Estimator (TPE) compounds by refitting its surrogate model after every trial. This thesis investigates whether a surrogate-free, coordinatewise search that adaptively weights hyperparameters by their contribution to performance can match or exceed BO's quality at lower cost, and proposes Adaptive Coordinate Random Search (ACRS) to address this question. Benchmarked against BO, Random Search (RS), and CMA-ES on a Multilayer Perceptron (MLP) and a Random Forest (RF) across multiple tabular binary classification datasets and seeds, ACRS matches or exceeds BO's predictive quality for both models, with differences within the across-seed standard deviation, while reducing runtime and energy consumption by roughly 10–27% relative to BO, depending on model type. ACRS thus offers a resource-efficient, surrogate-free alternative to BO.

## Project Structure

```
.
├── config.py             # Seeds, datasets, search spaces, ACRS parameters, trial budgets
├── data.py               # Dataset loading (OpenML/scikit-learn), subsampling, splits, scaling
├── models.py             # MLP (PyTorch) and Random Forest wrapper
├── train.py              # Training loops for MLP and RF
├── optimize.py           # ACRS implementation and Optuna-based BO (TPE), RS, CMA-ES
├── run_single.py         # Runs one combination of dataset × model × optimizer × seed
├── main.py               # Runs the full benchmark in parallel and merges results
├── plots.py              # Generates all figures and LaTeX tables from the result CSVs
├── data_cache/           # Cached datasets (.npz), downloaded from OpenML on first use
├── results/
│   ├── runs.csv          # One row per run (final metrics, duration, energy)
│   ├── trials.csv        # One row per trial (per-trial metrics and configs)
│   ├── plots/            # Generated figures
│   └── latex/            # Generated LaTeX tables
└── bachelor-thesis.pdf   # Thesis
```

## Getting Started

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the full benchmark

```bash
python main.py
```

This executes all 700 runs (7 optimizers × 2 models × 5 datasets × 10 seeds) in parallel subprocesses and merges the results into `results/runs.csv` and `results/trials.csv`. All datasets are included as cached `.npz` files in `data_cache/`, so no download is required. Missing datasets would be fetched from OpenML automatically. Note that the full benchmark takes several hours on CPU.

### Run a single experiment

```bash
python run_single.py --dataset <dataset> --model <model> --optimizer <optimizer> --seed <seed>
```

- Available datasets: `breast_cancer`, `diabetes`, `bank_marketing`, `german_credit`, `adult_income`.
- Available models: `mlp`, `rf`.
- Available optimizers: `bo`, `rs`, `cmaes`, `acrs`, `acrs_normal`, `acrs_shrink`, `acrs_normal_shrink`.
- Available seeds: `42`, `123`, `456`, `789`, `1234`, `2345`, `3456`, `4567`, `5678`, `6789`.

### Generate figures and tables

```bash
python plots.py
```

Reads the result CSVs and writes all figures to `results/plots/` and all LaTeX tables to `results/latex/`.
