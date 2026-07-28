# run_30_splits_phase3C.py
"""Phase 3-C: Ant Colony Optimization (ACO) feature selection + 11 classifiers."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from run_phase3_common import load_and_encode, get_or_create_splits, run_evaluation, print_results
from src.optimizers.aco_selector import ACOFeatureSelector

def main():
    X, y = load_and_encode()
    splits = get_or_create_splits(y)

    print("\n[+] Phase 3-C: Running ACO Feature Selection...")
    t0 = time.time()
    selector = ACOFeatureSelector(n_ants=30, iterations=10, random_state=42)
    mask = selector.select_features(X, y)
    np.save('features_opt_aco.npy', mask)
    print(f"[+] ACO selected {mask.sum()}/{len(mask)} features in {time.time()-t0:.1f}s")

    X_opt = X[:, mask]
    print(f"[+] Optimized feature matrix: {X_opt.shape}")

    print("\n[+] Running 30-split x 10-fold CV on all 11 algorithms (PARALLEL)...")
    t1 = time.time()
    results = run_evaluation(X_opt, y, splits)
    elapsed = time.time() - t1

    print_results("Phase 3-C (ACO)", "Ant Colony Optimization", results,
                  "results/phase3C_aco_results.csv", elapsed)

if __name__ == "__main__":
    main()
