import os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessor import TextPreprocessor

def get_algorithms():
    return {
        "SVM RBF":            SVC(kernel='rbf', C=10, gamma='scale'),
        "SVM Linear":         SVC(kernel='linear', C=10),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=3)":          KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)":          KNeighborsClassifier(n_neighbors=7),
        "Random Forest":      RandomForestClassifier(n_estimators=100, n_jobs=1),
        "AdaBoost":           AdaBoostClassifier(n_estimators=50),
        "Decision Tree":      DecisionTreeClassifier(),
        "Multinomial NB":     MultinomialNB(),
        "Naive Bayes":        GaussianNB(),
    }

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

class TextToDNAEncoder:
    def __init__(self, n_gram=3, max_features=98):
        self.codon_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n_gram, n_gram), max_features=max_features, sublinear_tf=True)
        self.word_to_base = {}
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

df = pd.read_csv('data/Promise_Dataset.csv')
y = df['Type'].apply(dna_mapping_5class).values
texts = df['Requirement'].tolist()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_pairs = list(skf.split(texts, y))

tf_vectorizer = TfidfVectorizer(max_features=482, sublinear_tf=True)
X_phase0 = tf_vectorizer.fit_transform(texts).toarray()

print('--- Phase 0 (TF-IDF Baseline) All Algorithms (10-Fold CV) ---')
algorithms = get_algorithms()
scaler = MinMaxScaler()
X_p0_scaled = scaler.fit_transform(X_phase0)

for name, clf in algorithms.items():
    X_cv = X_p0_scaled if name == "Multinomial NB" else X_phase0
    accs = []
    for tr, val in fold_pairs:
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_cv[tr], y[tr])
        accs.append(accuracy_score(y[val], clf_copy.predict(X_cv[val])))
    print(f'{name:25}: {np.mean(accs)*100:.2f}%')

print('\n--- Phase 2 (DNA+SBERT) All Algorithms (10-Fold CV) ---')
enc = TextToDNAEncoder()
X_c = enc.fit_transform(texts, y)
X_s = SentenceTransformer('all-MiniLM-L6-v2').encode(texts, show_progress_bar=False)
X_phase2 = np.hstack((X_c, X_s * 1.5))

algorithms2 = get_algorithms()
X_p2_scaled = scaler.fit_transform(X_phase2)

for name, clf in algorithms2.items():
    X_cv = X_p2_scaled if name == "Multinomial NB" else X_phase2
    accs = []
    for tr, val in fold_pairs:
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_cv[tr], y[tr])
        accs.append(accuracy_score(y[val], clf_copy.predict(X_cv[val])))
    print(f'{name:25}: {np.mean(accs)*100:.2f}%')

