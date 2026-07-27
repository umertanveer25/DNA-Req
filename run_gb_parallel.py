import os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
import multiprocessing

SEED, NUM_SPLITS, N_FOLDS = 42, 30, 10

print("[+] Loading Cached Features & Labels...", flush=True)
X = np.load('results/X_features_cache.npy')
y = np.load('results/y_labels_cache.npy', allow_pickle=True)

def eval_split(s):
    clf = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + s)
    accs, f1s = [], []
    for tr, val in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        bp = clf.predict(X[val])
        accs.append(accuracy_score(y[val], bp))
        f1s.append(f1_score(y[val], bp, average='macro', zero_division=0))
    return accs, f1s

if __name__ == '__main__':
    print("[+] Starting Parallel Gradient Boosting Evaluation...", flush=True)
    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(eval_split)(s) for s in range(NUM_SPLITS)
    )
    
    all_accs, all_f1s = [], []
    for a, f in results:
        all_accs.extend(a)
        all_f1s.extend(f)

    mean_acc = np.mean(all_accs) * 100
    mean_f1 = np.mean(all_f1s) * 100
    print("\n" + "="*60, flush=True)
    print(f"Gradient Boosting Baseline (300 Folds): Accuracy = {mean_acc:.2f}% | Macro F1 = {mean_f1:.2f}%", flush=True)
    print("="*60, flush=True)
