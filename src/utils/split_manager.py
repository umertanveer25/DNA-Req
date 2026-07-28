# src/utils/split_manager.py
"""Utility to generate or load predefined StratifiedKFold split indices.

The first time the baseline Phase 3 script runs, it will create a pickle file
`splits.pkl` containing a list of `(train_idx, val_idx)` tuples for each of the
30 random seeds. Subsequent runs (including the new optimizer pipelines) will
load this file so that *exactly the same* data partitions are used, ensuring a
fair comparison.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

SPLIT_FILE = "splits.pkl"


def generate_splits(y, n_splits=30, n_folds=10, random_state=42):
    """Generate `n_splits` different StratifiedKFold split sets.

    Returns a list of length `n_splits`; each element is a list of `(train, val)`
    index arrays for the `n_folds` folds.
    """
    rng = np.random.RandomState(random_state)
    split_sets = []
    for i in range(n_splits):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.randint(0, 1e6)))
        split_sets.append(list(skf.split(np.zeros_like(y), y)))
    return split_sets


def save_splits(split_sets, path=SPLIT_FILE):
    with open(path, "wb") as f:
        pickle.dump(split_sets, f)


def load_splits(path=SPLIT_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def get_or_create_splits(y, n_splits=30, n_folds=10, random_state=42):
    splits = load_splits()
    if splits is None:
        splits = generate_splits(y, n_splits, n_folds, random_state)
        save_splits(splits)
    return splits
