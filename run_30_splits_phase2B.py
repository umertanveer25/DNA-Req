# run_30_splits_phase2B.py
"""Phase 2-B: Particle Swarm Optimization (PSO) feature selection + 11 classifiers."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from run_phase2_common import load_and_encode, get_or_create_splits, run_evaluation, print_results
from src.optimizers.pso_selector import PSOFeatureSelector

def main():
    X, y = load_and_encode()
    splits = get_or_create_splits(y)

    print("\n[+] Phase 2-B: Running PSO Feature Selection...")
    t0 = time.time()
    selector = PSOFeatureSelector(swarm_size=30, iterations=10, random_state=42)
    mask = selector.select_features(X, y)
    np.save('features_opt_pso.npy', mask)
    print(f"[+] PSO selected {mask.sum()}/{len(mask)} features in {time.time()-t0:.1f}s")

    X_opt = X[:, mask]
    print(f"[+] Optimized feature matrix: {X_opt.shape}")

    print("\n[+] Running 30-split x 10-fold CV on all 11 algorithms (PARALLEL)...")
    t1 = time.time()
    results = run_evaluation(X_opt, y, splits)
    elapsed = time.time() - t1

    print_results("Phase 2-B (PSO)", "Particle Swarm Optimization", results,
                  "results/phase2B_pso_results.csv", elapsed)

if __name__ == "__main__":
    main()
