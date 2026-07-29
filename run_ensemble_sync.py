import os, sys, pickle, time
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
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
            analyzer='char', ngram_range=(n_gram, n_gram), max_features=max_features, sublinear_tf=True
        )

    def fit_transform(self, texts, labels):
        from collections import defaultdict
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        for text, label in zip(texts, labels):
            if label == 'N': continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base[w] = 'N'
                continue
            max_class, max_freq = 'N', 0
            for cls in ['A', 'T', 'C', 'G']:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            self.word_to_base[w] = max_class if (max_freq / total) >= 0.5 else 'N'
        
        seqs = []
        for text in texts:
            words = TextPreprocessor.clean_text(text).split()
            seq = "".join([self.word_to_base.get(w, 'N') for w in words])
            seqs.append(seq if seq else "N")
        
        return self.codon_vectorizer.fit_transform(seqs).toarray()


def get_ensemble_algorithms():
    return {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

def run_single_split(fold_pairs, X, y):
    algorithms = get_ensemble_algorithms()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    split_res = {}
    for name, clf in algorithms.items():
        accs = []
        for tr, val in fold_pairs:
            clf.fit(X_scaled[tr], y[tr])
            preds = clf.predict(X_scaled[val])
            accs.append(accuracy_score(y[val], preds))
        split_res[name] = np.mean(accs) * 100
    return split_res

if __name__ == '__main__':
    df = pd.read_csv('data/Promise_Dataset.csv')
    y = df['Type'].apply(dna_mapping_5class).values
    texts = df['Requirement'].tolist()

    with open('splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    first_split = splits[0] 

    tf_vectorizer = TfidfVectorizer(max_features=482, sublinear_tf=True)
    X_phase0 = tf_vectorizer.fit_transform(texts).toarray()
    
    res_phase0 = run_single_split(first_split, X_phase0, y)
    print('\n--- Phase 0 (TF-IDF) Ensemble Results ---')
    for k, v in res_phase0.items():
        print(f'{k:20}: {v:.2f}%')

    enc = TextToDNAEncoder(n_gram=3, max_features=98)
    X_codons = enc.fit_transform(texts, y)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_sbert = sbert.encode(texts, show_progress_bar=False)
    X_phase2 = np.hstack((X_codons, X_sbert * 1.5))
    
    res_phase2 = run_single_split(first_split, X_phase2, y)
    print('\n--- Phase 2 (DNA+SBERT) Ensemble Results ---')
    for k, v in res_phase2.items():
        print(f'{k:20}: {v:.2f}%')
