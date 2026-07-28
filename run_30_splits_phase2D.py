# run_30_splits_phase2D.py
"""Phase 2-D: Artificial Bee Colony (ABC) feature selection + 11 classifiers."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from run_phase2_common import load_and_encode, get_or_create_splits, run_evaluation, print_results
from src.optimizers.abc_selector import ABCFeatureSelector

def main():
    X, y = load_and_encode()
    splits = get_or_create_splits(y)

    print("\n[+] Phase 2-D: Running ABC Feature Selection...")
    t0 = time.time()
    selector = ABCFeatureSelector(colony_size=30, iterations=10, random_state=42)
    mask = selector.select_features(X, y)
    np.save('features_opt_abc.npy', mask)
    print(f"[+] ABC selected {mask.sum()}/{len(mask)} features in {time.time()-t0:.1f}s")

    X_opt = X[:, mask]
    print(f"[+] Optimized feature matrix: {X_opt.shape}")

    print("\n[+] Running 30-split x 10-fold CV on all 11 algorithms (PARALLEL)...")
    t1 = time.time()
    results = run_evaluation(X_opt, y, splits)
    elapsed = time.time() - t1

    print_results("Phase 2-D (ABC)", "Artificial Bee Colony", results,
                  "results/phase2D_abc_results.csv", elapsed)

if __name__ == "__main__":
    main()
