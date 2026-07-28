# src/run_30_splits_phase2A.py
"""Run Phase 3‑A (Genetic Algorithm feature selection) on all 11 classifiers.
This script uses the same 30‑split, 10‑fold CV partitions as the baseline
by loading `splits.pkl` via `src.utils.split_manager`.
The optimized feature mask is saved to `features_opt_ga.npy` for reproducibility.
"""

import os
import numpy as np

# Adjust imports for execution within src directory
# Adjust imports for package execution
from .phase2_base import load_data, encode_features, run_evaluation
from .utils.split_manager import get_or_create_splits
from .optimizers.ga_selector import GAFeatureSelector
from .models import get_classifier_dict

def main():
    # 1. Load raw data
    X_raw, y = load_data()
    # 2. Encode full feature matrix (482 dimensions)
    X = encode_features(X_raw, y)

    # 3. Obtain consistent splits (splits.pkl will be created/loaded)
    splits = get_or_create_splits(y)
    # The run_evaluation function expects a classifier dict; we reuse the same dict as baseline.
    algorithms = get_classifier_dict()

    # 4. GA feature selection
    selector = GAFeatureSelector()
    mask = selector.select_features(X, y)
    # Save mask for later reference
    np.save('features_opt_ga.npy', mask)
    X_opt = X[:, mask]

    # 5. Run evaluation using the stored splits
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from joblib import Parallel, delayed

    def run_split(fold_idx, fold_pair):
        train_idx, val_idx = fold_pair
        results = {}
        for name, clf in algorithms.items():
            X_use = X_opt if name != "Multinomial NB" else MinMaxScaler().fit_transform(X_opt)
            clf.fit(X_use[train_idx], y[train_idx])
            preds = clf.predict(X_use[val_idx])
            results[name] = accuracy_score(y[val_idx], preds) * 100
        return results

    all_results = {name: [] for name in algorithms}
    for outer_idx, fold_pairs in enumerate(splits):
        fold_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(run_split)(i, fold_pairs[i]) for i in range(len(fold_pairs))
        )
        for fr in fold_results:
            for name, acc in fr.items():
                all_results[name].append(acc)

    final_scores = {name: np.mean(accs) for name, accs in all_results.items()}

    # Print results
    print("\nAlgorithm                 | Phase 2‑A (GA) Avg Accuracy")
    print("=" * 55)
    for name, score in final_scores.items():
        print(f"{name:<25} | {score:6.2f}%")

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(list(final_scores.items()), columns=["Algorithm", "GA_Accuracy"])
    df.to_csv('phase2A_results.csv', index=False)

if __name__ == "__main__":
    main()
