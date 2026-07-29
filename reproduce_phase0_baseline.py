import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Guarantee reproducible results
np.random.seed(42)

print("="*80)
print("Phase 0 (Baseline) Reproducibility Script")
print("Model: TF-IDF (1000 features) + Classifiers")
print("Evaluation: 10-Fold CV across 30 Randomized Splits (300 total folds)")
print("="*80)

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

print("\n[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)

y = df['Target'].values
texts = df['Requirement'].tolist()

print("[+] Extracting TF-IDF Features (Phase 0)...")
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(texts).toarray()
X = MinMaxScaler().fit_transform(X)
print(f"[+] Final Feature Matrix Shape: {X.shape}")

# Define algorithms to test
algorithms = {
    'SVM RBF': SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM Linear': SVC(kernel='linear', C=1.0, random_state=42)
}

def evaluate_split(split_idx):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
    split_results = {algo: [] for algo in algorithms}
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        for algo_name, clf in algorithms.items():
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            split_results[algo_name].append(accuracy_score(y_val, preds))
            
    # Return average accuracy for this split for each algorithm
    return {algo: np.mean(accs) * 100 for algo, accs in split_results.items()}

print(f"\n[+] Running {len(algorithms)} algorithms across 30 Splits (PARALLEL MODE)...")
results = Parallel(n_jobs=-1, verbose=5)(delayed(evaluate_split)(i) for i in range(30))

print("\n"+"="*80)
print("FINAL RESULTS (30-Split Averages)")
print("="*80)

final_averages = {}
for algo in algorithms.keys():
    avg = np.mean([res[algo] for res in results])
    final_averages[algo] = avg

# Sort by accuracy
sorted_algos = sorted(final_averages.items(), key=lambda x: x[1], reverse=True)

for algo, acc in sorted_algos:
    print(f"{algo:<25} | {acc:>10.2f}%")
print("="*80)
