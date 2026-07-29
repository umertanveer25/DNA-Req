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
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Evaluating Phase 3 WITHOUT DNA (TF-IDF 98 Dims + Domain-Adapted SBERT)")
print("="*80)

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)

y = df['Target'].values
texts = df['Requirement'].tolist()

print("[+] Extracting standard TF-IDF features (exactly 98 dims to match DNA)...")
tfidf = TfidfVectorizer(max_features=98)
X_tfidf = tfidf.fit_transform(texts).toarray()

print("[+] Loading Domain-Adapted SBERT (384 dims)...")
sbert_finetuned = SentenceTransformer('models/sbert-promise-finetuned')
X_sbert = sbert_finetuned.encode(texts, show_progress_bar=False)

# Fuse TF-IDF + SBERT (No DNA)
X = np.hstack((X_tfidf, X_sbert * 1.5))
X = MinMaxScaler().fit_transform(X)
print(f"[+] Final Matrix Shape (NO DNA): {X.shape}")

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
            
    return {algo: np.mean(accs) * 100 for algo, accs in split_results.items()}

print("\n[+] Running Top 3 algorithms across 30 Splits (PARALLEL)...")
results = Parallel(n_jobs=-1, verbose=5)(delayed(evaluate_split)(i) for i in range(30))

print("\n"+"="*80)
print("FINAL RESULTS WITHOUT DNA (98 Dims TF-IDF) (30-Split Averages)")
print("="*80)

for algo in algorithms.keys():
    avg = np.mean([res[algo] for res in results])
    print(f"{algo:<25} | {avg:>10.2f}%")
print("="*80)
