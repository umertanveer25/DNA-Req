# src/phase2_base.py
"""Common utilities for Phase 3 (codon + SBERT) pipelines.

This module provides:
- `load_data()` – loads the PROMISE dataset and returns feature matrices and labels.
- `encode_features(requirements, targets)` – runs the TextToDNAEncoder and SBERT encoder and returns the fused 482‑dim matrix.
- `run_evaluation(X, y, algorithms, n_splits=30)` – performs the parallel 30‑split, 10‑fold CV evaluation used by all Phase 3 scripts.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer
from .preprocessor import TextPreprocessor
from .features import TextToDNAEncoder

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load PROMISE dataset and prepare labels.

    Returns
    -------
    X_raw : list[str]
        List of requirement texts.
    y : np.ndarray
        Array of 5‑class target labels (A, T, C, G, N).
    """
    df = pd.read_csv('data/Promise_Dataset.csv')
    df['Type'] = df['Type'].str.strip()
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    df['Target'] = df['Type'].apply(lambda x: mapping.get(str(x).strip().upper(), 'N'))
    X_raw = df['Requirement'].tolist()
    y = df['Target'].values
    return X_raw, y

# ---------------------------------------------------------------------------
# Feature encoding (codon frequencies + SBERT)
# ---------------------------------------------------------------------------

def encode_features(requirements, targets):
    """Encode raw requirement texts into the 482‑dim fusion matrix.

    Parameters
    ----------
    requirements : list[str]
        Raw requirement sentences.
    targets : list[str]
        Corresponding target labels (used only for the DNA encoder fit).

    Returns
    -------
    X : np.ndarray
        Fusion matrix of shape (n_samples, 482).
    """
    dna_encoder = TextToDNAEncoder(n_gram=3, max_features=98)
    X_codons = dna_encoder.fit_transform(requirements, targets)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_sbert = sbert.encode(requirements, show_progress_bar=False)
    X = np.hstack((X_codons, X_sbert * 1.5))
    return X

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _run_split(split_idx, X, y, algorithms):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
    split_res = {}
    for name, clf in algorithms.items():
        X_cv = X if name != "Multinomial NB" else MinMaxScaler().fit_transform(X)
        accs = []
        try:
            for tr, val in skf.split(X_cv, y):
                clf.fit(X_cv[tr], y[tr])
                preds = clf.predict(X_cv[val])
                accs.append(accuracy_score(y[val], preds))
            split_res[name] = np.mean(accs) * 100
        except Exception:
            split_res[name] = 0.0
    return split_res


def run_evaluation(X, y, algorithms, n_splits=30):
    """Run the 30‑split, 10‑fold CV evaluation in parallel.

    Returns a dict mapping algorithm name → average accuracy (percentage).
    """
    results_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(_run_split)(i, X, y, algorithms) for i in range(n_splits)
    )
    final_results = {name: [] for name in algorithms}
    for res in results_list:
        for name, acc in res.items():
            final_results[name].append(acc)
    return {name: np.mean(accs) for name, accs in final_results.items()}
