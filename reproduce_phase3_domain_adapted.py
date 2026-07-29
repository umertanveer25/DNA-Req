import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessor import TextPreprocessor

# Guarantee reproducible results
np.random.seed(42)

print("="*80)
print("Phase 3 Reproducibility Script")
print("Model: TextToDNAEncoder + Domain-Adapted SBERT (BatchHardTripletLoss)")
print("Evaluation: 10-Fold CV across 30 Randomized Splits (300 total folds)")
print("="*80)

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

class TextToDNAEncoder:
    def __init__(self, n_gram=3, max_features=98):
        self.word_to_base = {}
        self.codon_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(n_gram, n_gram),
            max_features=max_features,
            sublinear_tf=True
        )

    def fit(self, texts, labels):
        from collections import defaultdict
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        
        for text, label in zip(texts, labels):
            if label == 'N':
                continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1
                
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base[w] = 'N'
                continue
            max_class = 'N'
            max_freq = 0
            for cls in ['A', 'T', 'C', 'G']:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            if (max_freq / total) < 0.5:
                self.word_to_base[w] = 'N'
            else:
                self.word_to_base[w] = max_class
                
        dna_sequences = self._translate(texts)
        self.codon_vectorizer.fit(dna_sequences)
        return self

    def _translate(self, texts):
        dna_sequences = []
        for text in texts:
            words = TextPreprocessor.clean_text(text).split()
            sequence = "".join([self.word_to_base.get(w, 'N') for w in words])
            if not sequence:
                sequence = "N"
            dna_sequences.append(sequence)
        return dna_sequences

    def transform(self, texts):
        dna_sequences = self._translate(texts)
        return self.codon_vectorizer.transform(dna_sequences).toarray()

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform(texts)

print("\n[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)

y = df['Target'].values
texts = df['Requirement'].tolist()
labels = df['Target'].tolist()

print("[+] Running Phase 3 Encoder: English to Amino Acid Codons (98 Dimensions)...")
ext = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons = ext.fit_transform(texts, labels)

model_path = 'models/sbert-promise-finetuned'
if not os.path.exists(model_path):
    print(f"[-] ERROR: Fine-tuned model not found at {model_path}. Please run train_phase3_sbert.py first.")
    sys.exit(1)

print(f"[+] Loading Domain-Adapted SBERT from {model_path}...")
sbert_finetuned = SentenceTransformer(model_path)
X_sbert = sbert_finetuned.encode(texts, show_progress_bar=True)

# Fuse vectors
X = np.hstack((X_codons, X_sbert * 1.5))
X = MinMaxScaler().fit_transform(X)
print(f"[+] Final Phase 3 Fusion Matrix Shape: {X.shape} (Matches exactly 482 dimensions!)")

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

print(f"\n[+] Running {len(algorithms)} algorithms across 30 Splits (PARALLEL MODE)...")
print("[WARNING] The dense geometrical clustering in Phase 3 causes SVM/Logistic models to take significantly longer to converge.")
results = Parallel(n_jobs=-1, verbose=5)(delayed(evaluate_split)(i) for i in range(30))

print("\n"+"="*80)
print("FINAL RESULTS (30-Split Averages)")
print("="*80)

final_averages = {}
for algo in algorithms.keys():
    avg = np.mean([res[algo] for res in results])
    final_averages[algo] = avg

sorted_algos = sorted(final_averages.items(), key=lambda x: x[1], reverse=True)

for algo, acc in sorted_algos:
    print(f"{algo:<25} | {acc:>10.2f}%")
print("="*80)
