import os, sys, multiprocessing, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed

os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

SEED, NUM_SPLITS, N_FOLDS = 42, 30, 10
cache_X = 'results/X_features_cache.npy'
cache_y = 'results/y_labels_cache.npy'

# Ensure features are present
if not os.path.exists(cache_X) or not os.path.exists(cache_y):
    print("[+] Extracting and caching features for Gradient Boosting benchmark...")
    df = pd.read_csv('data/Promise_Dataset.csv')
    df['Type'] = df['Type'].str.strip()
    df['DNA_Target'] = df['Type'].apply(dna_mapping)
    ext = DNAFeatureExtractor(max_tfidf_features=50)
    X = ext.fit_transform(df['Requirement'].tolist())
    y = df['DNA_Target'].values
    os.makedirs('results', exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_y, y)
else:
    print("[+] Loading Cached Features & Labels...", flush=True)
    X = np.load(cache_X)
    y = np.load(cache_y, allow_pickle=True)

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
    results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(eval_split)(s) for s in range(NUM_SPLITS)
    )
    
    all_accs, all_f1s = [], []
    for a, f in results:
        all_accs.extend(a)
        all_f1s.extend(f)

    mean_acc = np.mean(all_accs) * 100
    mean_f1 = np.mean(all_f1s) * 100
    print("\n" + "=" * 60, flush=True)
    print(f"Gradient Boosting Baseline (300 Folds): Accuracy = {mean_acc:.2f}% | Macro F1 = {mean_f1:.2f}%", flush=True)
    print("=" * 60, flush=True)
