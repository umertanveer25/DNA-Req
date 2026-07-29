import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessor import TextPreprocessor

print("="*80)
print("CLASS-WISE ACCURACY COMPARISON (WITH DNA vs WITHOUT DNA)")
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

df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)
y = df['Target'].values
texts = df['Requirement'].tolist()
labels = df['Target'].tolist()

print("[+] Loading Domain-Adapted SBERT (384 dims)...")
sbert_finetuned = SentenceTransformer('models/sbert-promise-finetuned')
X_sbert = sbert_finetuned.encode(texts, show_progress_bar=False)

print("[+] Extracting WITHOUT DNA (TF-IDF 98 dims)...")
tfidf = TfidfVectorizer(max_features=98)
X_tfidf = tfidf.fit_transform(texts).toarray()
X_no_dna = np.hstack((X_tfidf, X_sbert * 1.5))
X_no_dna = MinMaxScaler().fit_transform(X_no_dna)

print("[+] Extracting WITH DNA (Codons 98 dims)...")
ext = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons = ext.fit_transform(texts, labels)
X_with_dna = np.hstack((X_codons, X_sbert * 1.5))
X_with_dna = MinMaxScaler().fit_transform(X_with_dna)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf = SVC(kernel='linear', C=1.0, random_state=42) # Use linear for fast execution to prove class differences

y_true_all = []
y_pred_no_dna = []
y_pred_with_dna = []

print("[+] Running 10-Fold CV...")
for tr_idx, val_idx in skf.split(X_no_dna, y):
    y_val = y[val_idx]
    y_true_all.extend(y_val)
    
    # Train NO DNA
    clf.fit(X_no_dna[tr_idx], y[tr_idx])
    y_pred_no_dna.extend(clf.predict(X_no_dna[val_idx]))
    
    # Train WITH DNA
    clf.fit(X_with_dna[tr_idx], y[tr_idx])
    y_pred_with_dna.extend(clf.predict(X_with_dna[val_idx]))

print("\n"+"="*80)
print("WITHOUT DNA (TF-IDF 98 + SBERT) - CLASS REPORT")
print("="*80)
print(classification_report(y_true_all, y_pred_no_dna, digits=4))

print("\n"+"="*80)
print("WITH DNA (Codons 98 + SBERT) - CLASS REPORT")
print("="*80)
print(classification_report(y_true_all, y_pred_with_dna, digits=4))

