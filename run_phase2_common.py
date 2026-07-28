# run_phase2_common.py
"""Shared utilities for Phase 2-A through 3-F optimizer comparison scripts.

This module provides:
  - load_and_encode()  - loads PROMISE dataset, encodes to 482-dim fusion matrix
  - ALGORITHMS         - the exact 11 classifiers from the baseline Phase 2 script
  - get_or_create_splits() - reproducible 30-split x 10-fold CV partitions
  - run_evaluation()   - parallel 30-split evaluation returning per-algorithm averages
  - print_results()    - formatted table output + CSV export + baseline comparison
"""

import os, sys, time, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# Baseline accuracy floor (SVM RBF must stay >= this)
# ---------------------------------------------------------------------------
BASELINE_SVM_RBF = 86.61  # from baseline_output.txt

# ---------------------------------------------------------------------------
# The exact 11 algorithms from run_30_splits_phase2.py (baseline)
# ---------------------------------------------------------------------------
def get_algorithms():
    """Return fresh (unfitted) copies of the 11 baseline classifiers."""
    return {
        "SVM RBF":            SVC(kernel='rbf', C=10, gamma='scale'),
        "SVM Linear":         SVC(kernel='linear', C=10),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=3)":          KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)":          KNeighborsClassifier(n_neighbors=7),
        "Random Forest":      RandomForestClassifier(n_estimators=100, n_jobs=1),
        "AdaBoost":           AdaBoostClassifier(n_estimators=50),
        "Decision Tree":      DecisionTreeClassifier(),
        "Multinomial NB":     MultinomialNB(),
        "Naive Bayes":        GaussianNB(),
    }

# ---------------------------------------------------------------------------
# DNA encoding (identical to baseline)
# ---------------------------------------------------------------------------
def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')


class TextToDNAEncoder:
    def __init__(self, n_gram=3, max_features=98):
        self.word_to_base = {}
        self.codon_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(n_gram, n_gram),
            max_features=max_features,
            sublinear_tf=True
        )

    def fit(self, texts, labels):
        from collections import defaultdict
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        for text, label in zip(texts, labels):
            if label == 'N':
                continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base[w] = 'N'
                continue
            max_class, max_freq = 'N', 0
            for cls in ['A', 'T', 'C', 'G']:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            self.word_to_base[w] = max_class if (max_freq / total) >= 0.5 else 'N'
        dna_sequences = self._translate(texts)
        self.codon_vectorizer.fit(dna_sequences)
        return self

    def _translate(self, texts):
        seqs = []
        for text in texts:
            words = TextPreprocessor.clean_text(text).split()
            seq = "".join([self.word_to_base.get(w, 'N') for w in words])
            seqs.append(seq if seq else "N")
        return seqs

    def transform(self, texts):
        return self.codon_vectorizer.transform(self._translate(texts)).toarray()

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform(texts)


# ---------------------------------------------------------------------------
# Data loading & feature encoding
# ---------------------------------------------------------------------------
def load_and_encode():
    """Load PROMISE dataset, encode to 482-dim fusion matrix.

    Returns (X, y) where X.shape == (969, 482).
    """
    print("[+] Loading PROMISE dataset...")
    df = pd.read_csv('data/Promise_Dataset.csv')
    df['Type'] = df['Type'].str.strip()
    df['Target'] = df['Type'].apply(dna_mapping_5class)
    y = df['Target'].values

    print("[+] Encoding: English -> Amino Acid Codons (98 dims)...")
    enc = TextToDNAEncoder(n_gram=3, max_features=98)
    X_codons = enc.fit_transform(df['Requirement'].tolist(), df['Target'].tolist())

    print("[+] Encoding: Deep Semantic DNA SBERT (384 dims)...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_sbert = sbert.encode(df['Requirement'].tolist(), show_progress_bar=False)

    X = np.hstack((X_codons, X_sbert * 1.5))
    print(f"[+] Final Phase 2 Fusion Matrix: {X.shape} (482 dimensions)")
    return X, y


# ---------------------------------------------------------------------------
# Split management (reproducible partitions)
# ---------------------------------------------------------------------------
SPLIT_FILE = "splits.pkl"

def get_or_create_splits(y, n_splits=30, n_folds=10, seed=42):
    """Load or generate the canonical 30x10 CV partitions."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "rb") as f:
            splits = pickle.load(f)
        print(f"[+] Loaded stored splits from {SPLIT_FILE}")
        return splits

    print(f"[+] Generating {n_splits}x{n_folds} splits (seed={seed})...")
    rng = np.random.RandomState(seed)
    splits = []
    for _ in range(n_splits):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=int(rng.randint(0, 1_000_000)))
        splits.append(list(skf.split(np.zeros(len(y)), y)))
    with open(SPLIT_FILE, "wb") as f:
        pickle.dump(splits, f)
    print(f"[+] Saved splits -> {SPLIT_FILE}")
    return splits


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
def _run_single_split(split_idx, fold_pairs, X, y):
    """Run all 11 algorithms on one split's 10 folds."""
    algorithms = get_algorithms()  # fresh unfitted copies every split
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    split_res = {}
    for name, clf in algorithms.items():
        X_cv = X_scaled if name == "Multinomial NB" else X
        accs = []
        for tr, val in fold_pairs:
            try:
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_cv[tr], y[tr])
                preds = clf_copy.predict(X_cv[val])
                accs.append(accuracy_score(y[val], preds))
            except Exception:
                accs.append(0.0)
        split_res[name] = np.mean(accs) * 100
    return split_res


def run_evaluation(X, y, splits):
    """Run 30-split x 10-fold CV on all 11 algorithms in parallel.

    Returns dict: algorithm_name -> average accuracy (%)
    """
    results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run_single_split)(i, splits[i], X, y)
        for i in range(len(splits))
    )
    # Aggregate
    algo_names = list(get_algorithms().keys())
    aggregated = {name: [] for name in algo_names}
    for res in results_list:
        for name, acc in res.items():
            aggregated[name].append(acc)
    return {name: np.mean(accs) for name, accs in aggregated.items()}


# ---------------------------------------------------------------------------
# Results display & persistence
# ---------------------------------------------------------------------------
def print_results(phase_name, optimizer_name, results, csv_path, elapsed):
    """Print a formatted table, save CSV, and check against baseline."""
    print(f"\n[+] Completed {phase_name} ({optimizer_name}) in {elapsed:.1f}s")
    print("=" * 70)
    print(f"{'Algorithm':<25} | {phase_name + ' Avg Accuracy':>30}")
    print("=" * 70)

    sorted_algs = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_algs:
        marker = " *" if name == "SVM RBF" else ""
        print(f"{name:<25} | {acc:>29.2f}%{marker}")
    print("=" * 70)

    # Baseline check
    svm_acc = results.get("SVM RBF", 0)
    if svm_acc >= BASELINE_SVM_RBF:
        print(f"[PASS] SVM RBF ({svm_acc:.2f}%) meets baseline floor ({BASELINE_SVM_RBF}%)")
    else:
        print(f"[WARN] SVM RBF ({svm_acc:.2f}%) is BELOW baseline floor ({BASELINE_SVM_RBF}%)")

    # Save CSV
    df = pd.DataFrame(list(results.items()), columns=["Algorithm", f"{phase_name}_Accuracy"])
    df = df.sort_values(f"{phase_name}_Accuracy", ascending=False)
    df.to_csv(csv_path, index=False)
    print(f"[+] Results saved -> {csv_path}")

    # Save JSON for programmatic comparison
    json_path = csv_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[+] JSON saved  -> {json_path}")

    return results
