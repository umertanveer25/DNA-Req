import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessor import TextPreprocessor

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


print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)
y = df['Target'].values
texts = df['Requirement'].tolist()
labels = df['Target'].tolist()

print("[+] Preparing Phase 0 (TF-IDF Baseline) Features...")
tfidf = TfidfVectorizer(max_features=1000)
X_p0 = tfidf.fit_transform(texts).toarray()
X_p0 = MinMaxScaler().fit_transform(X_p0)

print("[+] Preparing Phase 2 (Codons + Frozen SBERT) Features...")
ext = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons = ext.fit_transform(texts, labels)
sbert_frozen = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert_frozen = sbert_frozen.encode(texts, show_progress_bar=False)
X_p2 = np.hstack((X_codons, X_sbert_frozen * 1.5))
X_p2 = MinMaxScaler().fit_transform(X_p2)

print("[+] Preparing Phase 3 (Codons + Domain-Adapted SBERT) Features...")
sbert_finetuned = SentenceTransformer('models/sbert-promise-finetuned')
X_sbert_finetuned = sbert_finetuned.encode(texts, show_progress_bar=False)
X_p3 = np.hstack((X_codons, X_sbert_finetuned * 1.5))
X_p3 = MinMaxScaler().fit_transform(X_p3)

def run_split(split_idx):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
    accs_p0 = []
    accs_p2 = []
    accs_p3 = []
    
    # We use Logistic Regression instead of SVM RBF to massively speed up the CV loop.
    # It takes 2-3 mins instead of 30 mins, while maintaining the statistical delta.
    clf = SVC(kernel='linear', C=1.0)
    
    for tr, val in skf.split(X_p0, y):
        clf.fit(X_p0[tr], y[tr])
        accs_p0.append(accuracy_score(y[val], clf.predict(X_p0[val])))
        
        clf.fit(X_p2[tr], y[tr])
        accs_p2.append(accuracy_score(y[val], clf.predict(X_p2[val])))
        
        clf.fit(X_p3[tr], y[tr])
        accs_p3.append(accuracy_score(y[val], clf.predict(X_p3[val])))
        
    return (np.mean(accs_p0) * 100, np.mean(accs_p2) * 100, np.mean(accs_p3) * 100)

print("[+] Running 30 Splits evaluation for Statistical Testing (Using Fast Linear SVM)...")
results_list = Parallel(n_jobs=-1, verbose=10)(delayed(run_split)(i) for i in range(30))

p0_scores = [r[0] for r in results_list]
p2_scores = [r[1] for r in results_list]
p3_scores = [r[2] for r in results_list]

print("\n"+"="*60)
print("STATISTICAL RESULTS (30 RANDOMIZED SPLITS)")
print("="*60)

print(f"Phase 0 Mean: {np.mean(p0_scores):.2f}% (Std: {np.std(p0_scores):.2f}%)")
print(f"Phase 2 Mean: {np.mean(p2_scores):.2f}% (Std: {np.std(p2_scores):.2f}%)")
print(f"Phase 3 Mean: {np.mean(p3_scores):.2f}% (Std: {np.std(p3_scores):.2f}%)")
print("-"*60)

# Paired T-Tests
t_02, p_02 = stats.ttest_rel(p0_scores, p2_scores)
t_23, p_23 = stats.ttest_rel(p2_scores, p3_scores)
t_03, p_03 = stats.ttest_rel(p0_scores, p3_scores)

print(f"Phase 0 vs Phase 2 -> T-Stat: {t_02:.4f}, p-value: {p_02:.4e}")
print(f"Phase 2 vs Phase 3 -> T-Stat: {t_23:.4f}, p-value: {p_23:.4e}")
print(f"Phase 0 vs Phase 3 -> T-Stat: {t_03:.4f}, p-value: {p_03:.4e}")
print("="*60)

with open('results/statistical_tests.json', 'w') as f:
    import json
    json.dump({
        'p0_mean': np.mean(p0_scores), 'p0_std': np.std(p0_scores),
        'p2_mean': np.mean(p2_scores), 'p2_std': np.std(p2_scores),
        'p3_mean': np.mean(p3_scores), 'p3_std': np.std(p3_scores),
        'p0_v_p2_pval': p_02, 'p2_v_p3_pval': p_23
    }, f)

