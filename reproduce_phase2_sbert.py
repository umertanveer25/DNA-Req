import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.utils.seed_manager import set_global_seed, get_reproducible_classifiers
from src.evaluation_leakfree import evaluate_fold_leakfree_precomputed_sbert

set_global_seed(42)

print("="*80, flush=True)
print("Phase 2 Reproducibility Script [LEAK-FREE FAST PARALLEL]", flush=True)
print("="*80, flush=True)

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)

y = df['Target'].values
texts = df['Requirement'].tolist()

print("[+] Pre-computing Frozen SBERT Embeddings globally...", flush=True)
sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert_global = sbert.encode(texts, show_progress_bar=False)
print(f"[+] SBERT Shape: {X_sbert_global.shape}", flush=True)

algorithms = get_reproducible_classifiers(seed=42)
print(f"\n[+] Running {len(algorithms)} algorithms across 30 Splits (300 Folds) in PARALLEL...", flush=True)

def evaluate_algo(algo_name, clf):
    all_fold_accs = []
    for split_idx in range(30):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
        for train_idx, val_idx in skf.split(texts, y):
            acc = evaluate_fold_leakfree_precomputed_sbert(
                train_idx, val_idx, texts, X_sbert_global, y, clf
            )
            all_fold_accs.append(acc)
    # Return average accuracy * 100
    return algo_name, np.mean(all_fold_accs) * 100

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_algo)(name, clf) for name, clf in algorithms.items()
)

print("\n" + "="*80, flush=True)
print("FINAL TRUE RESULTS (30-Split Averages, Zero Leakage)", flush=True)
print("="*80, flush=True)

sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for algo, acc in sorted_results:
    print(f"{algo:<25} | {acc:>10.2f}%", flush=True)
print("="*80, flush=True)


