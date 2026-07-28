# run_30_splits_phase2E.py
"""Phase 2-E: Grey Wolf Optimizer (GWO) feature selection + 11 classifiers."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from run_phase2_common import load_and_encode, get_or_create_splits, run_evaluation, print_results
from src.optimizers.gwo_selector import GWOFeatureSelector

def main():
    X, y = load_and_encode()
    splits = get_or_create_splits(y)

    print("\n[+] Phase 2-E: Running GWO Feature Selection...")
    t0 = time.time()
    selector = GWOFeatureSelector(pack_size=30, iterations=10, random_state=42)
    mask = selector.select_features(X, y)
    np.save('features_opt_gwo.npy', mask)
    print(f"[+] GWO selected {mask.sum()}/{len(mask)} features in {time.time()-t0:.1f}s")

    X_opt = X[:, mask]
    print(f"[+] Optimized feature matrix: {X_opt.shape}")

    print("\n[+] Running 30-split x 10-fold CV on all 11 algorithms (PARALLEL)...")
    t1 = time.time()
    results = run_evaluation(X_opt, y, splits)
    elapsed = time.time() - t1

    print_results("Phase 2-E (GWO)", "Grey Wolf Optimizer", results,
                  "results/phase2E_gwo_results.csv", elapsed)

if __name__ == "__main__":
    main()
