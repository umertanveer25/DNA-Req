import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from src.features import CanonicalDNAEncoder
import torch

sys.path.insert(0, '.')
from src.utils.seed_manager import get_reproducible_classifiers

print("="*80)
print("Phase 3 Reproducibility Script [TRUE LEAK-FREE DOMAIN ADAPTATION]")
print("Evaluation: 10-Fold CV (1 Split due to extreme GPU computation time)")
print("="*80)

# Load data
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
df['Target'] = df['Type'].apply(lambda x: mapping.get(str(x).upper(), 'N'))
y = df['Target'].values
texts = df['Requirement'].tolist()

algorithms = get_reproducible_classifiers()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {algo: [] for algo in algorithms}

for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, y)):
    print(f"\n[+] Running Fold {fold+1}/10...")
    
    X_text_tr = [texts[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_text_val = [texts[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    
    print("    [+] Fine-tuning SBERT strictly on Training Fold (No Data Leakage)...")
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    
    # Create training examples
    train_examples = []
    for i in range(len(X_text_tr)):
        for j in range(i+1, len(X_text_tr)):
            label = 1.0 if y_tr[i] == y_tr[j] else 0.0
            train_examples.append(InputExample(texts=[X_text_tr[i], X_text_tr[j]], label=label))
            
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train for 1 epoch to save time, on GPU
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, show_progress_bar=False)
    
    print("    [+] Generating Embeddings...")
    sbert_tr = model.encode(X_text_tr, show_progress_bar=False)
    sbert_val = model.encode(X_text_val, show_progress_bar=False)
    
    print("    [+] Building DNA Features...")
    dna_encoder = CanonicalDNAEncoder(n_gram=3, max_features=98)
    dna_encoder.fit(X_text_tr, y_tr)
    
    dna_tr = dna_encoder.transform(X_text_tr)
    dna_val = dna_encoder.transform(X_text_val)
    
    X_tr_final = np.hstack((dna_tr, sbert_tr * 1.5))
    X_val_final = np.hstack((dna_val, sbert_val * 1.5))
    
    print("    [+] Evaluating Classifiers...")
    for name, clf in algorithms.items():
        clf.fit(X_tr_final, y_tr)
        acc = clf.score(X_val_final, y_val)
        results[name].append(acc)

print("\n" + "="*80)
print("FINAL RESULTS (True Leak-Free 10-Fold Averages)")
print("="*80)
final_averages = {algo: np.mean(accs)*100 for algo, accs in results.items()}
for algo, acc in sorted(final_averages.items(), key=lambda x: x[1], reverse=True):
    print(f"{algo:<25} | {acc:>10.2f}%")
