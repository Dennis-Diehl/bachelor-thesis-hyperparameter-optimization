"""Konstanten, Suchräume und Seeds für den HPO-Benchmark."""

from ray import tune

# Reproduzierbarkeit
SEEDS = [42, 123, 456]

# Datensätze
DATASETS = ["breast_cancer", "diabetes", "bank_marketing"]

# Daten-Split
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # bezogen auf den Rest nach Test-Split → effektiv 60/20/20

# Anzahl Trials pro Optimierer-Lauf
N_TRIALS = 40

# MLP-Training
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# MLP-Suchraum
MLP_SEARCH_SPACE = {
    "learning_rate":  tune.loguniform(1e-5, 1e-1),
    "batch_size":     tune.choice([16, 32, 64, 128]),
    "hidden_dim":     tune.randint(32, 257),
    "dropout":        tune.uniform(0.0, 0.5),
    "num_layers":     tune.choice([1, 2, 3]),
    "optimizer_name": tune.choice(["adam", "sgd", "adamw"]),
    "weight_decay":   tune.loguniform(1e-6, 1e-2),
    "activation":     tune.choice(["relu", "tanh"]),
}

# Random-Forest-Suchraum
RF_SEARCH_SPACE = {
    "n_estimators":      tune.randint(50, 501),
    "max_depth":         tune.randint(3, 31),
    "min_samples_split": tune.randint(2, 21),
    "max_features":      tune.choice(["sqrt", "log2"]),
    "min_samples_leaf":  tune.randint(1, 11),
    "criterion":         tune.choice(["gini", "entropy"]),
    "max_samples":       tune.uniform(0.5, 1.0),
}
