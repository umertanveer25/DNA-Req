import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
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
        self.word_is_ambiguous = {}
        self.codon_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 1),
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
                
        rng = np.random.RandomState(42)
        canonical_bases = ['A', 'T', 'C', 'G']
        
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base[w] = rng.choice(canonical_bases)
                self.word_is_ambiguous[w] = True
                continue
            max_class = 'N'
            max_freq = 0
            for cls in canonical_bases:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            if (max_freq / total) < 0.5:
                self.word_to_base[w] = rng.choice(canonical_bases)
                self.word_is_ambiguous[w] = True
            else:
                self.word_to_base[w] = max_class
                self.word_is_ambiguous[w] = False
                
        codon_sequences = self._translate(texts)
        self.codon_vectorizer.fit(codon_sequences)
        return self

    def _translate(self, texts):
        rng = np.random.RandomState(42)
        canonical_bases = ['A', 'T', 'C', 'G']
        codon_sequences = []
        
        for text in texts:
            words = TextPreprocessor.clean_text(text).split()
            bases = []
            ambiguous_mask = []
            
            for w in words:
                if w in self.word_to_base:
                    bases.append(self.word_to_base[w])
                    ambiguous_mask.append(self.word_is_ambiguous[w])
                else:
                    bases.append(rng.choice(canonical_bases))
                    ambiguous_mask.append(True)
                    
            codons = []
            if len(bases) < 3:
                codons.append("NNN")
            else:
                for i in range(len(bases) - 2):
                    if any(ambiguous_mask[i:i+3]):
                        codons.append("NNN")
                    else:
                        codons.append("".join(bases[i:i+3]))
                        
            codon_sequences.append(" ".join(codons))
        return codon_sequences

    def transform(self, texts):
        codon_sequences = self._translate(texts)
        return self.codon_vectorizer.transform(codon_sequences).toarray()

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform(texts)

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)
y = df['Target'].values

print("[+] Running Phase 3 Encoder: English to Amino Acid Codons (98 Dimensions)...")
ext = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons = ext.fit_transform(df['Requirement'].tolist(), df['Target'].tolist())

print("[+] Adding Domain-Adapted Deep Semantic DNA SBERT (384 Dimensions)...")
sbert_model_path = "models/sbert-promise-finetuned"
if not os.path.exists(sbert_model_path):
    print(f"Error: {sbert_model_path} not found. Run train_phase3_sbert.py first.")
    sys.exit(1)
sbert = SentenceTransformer(sbert_model_path)
X_sbert = sbert.encode(df['Requirement'].tolist(), show_progress_bar=False)

X = np.hstack((X_codons, X_sbert * 1.5))
print(f"[+] Final Phase 3 Fusion Matrix Shape: {X.shape} (Matches exactly 482 dimensions!)")

algorithms = {
    "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale'),
    "SVM Linear": SVC(kernel='linear', C=10),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=1),
    "AdaBoost": AdaBoostClassifier(n_estimators=50),
    "Decision Tree": DecisionTreeClassifier(),
    "Multinomial NB": MultinomialNB(),
    "Naive Bayes": GaussianNB()
}

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def run_split(split_idx):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
    split_res = {}
    for name, clf in algorithms.items():
        X_cv = X_scaled if name == "Multinomial NB" else X
        accs = []
        try:
            for tr, val in skf.split(X_cv, y):
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_cv[tr], y[tr])
                preds = clf_copy.predict(X_cv[val])
                accs.append(accuracy_score(y[val], preds))
            split_res[name] = np.mean(accs) * 100
        except Exception:
            split_res[name] = 0.0
    return split_res

print("\\n[+] Running Phase 3 (Amino Acids + Domain-Adapted SBERT) 10-Fold CV across 30 Randomized Splits (PARALLEL MODE)...")
import time
start_time = time.time()

num_splits = 30
# Run 30 splits completely in parallel
results_list = Parallel(n_jobs=-1, verbose=10)(delayed(run_split)(i) for i in range(num_splits))

# Aggregate results
final_results = {name: [] for name in algorithms.keys()}
for res in results_list:
    for name, acc in res.items():
        final_results[name].append(acc)

print(f"\\n[+] Completed 30 Randomized Splits in {time.time() - start_time:.2f} seconds!", flush=True)
print("="*80, flush=True)
print(f"{'Algorithm':<25} | {'Phase 3 (30-Split Average)':<30}", flush=True)
print("="*80, flush=True)

final_averages = {}
for name in algorithms.keys():
    final_averages[name] = np.mean(final_results[name])

sorted_algs = sorted(final_averages.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_algs:
    print(f"{name:<25} | {acc:>25.2f}%", flush=True)
print("="*80, flush=True)

