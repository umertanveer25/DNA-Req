import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.features import dna_mapping, TextPreprocessor

class Phase1ExperimentalExtractor:
    def __init__(self, max_tfidf_features=50, ngram_range=(1,3), sbert_model_name='all-MiniLM-L6-v2'):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=ngram_range,
            sublinear_tf=True
        )

    def fit_transform(self, texts):
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts).toarray()
        sbert_embeddings = self.sbert_model.encode(cleaned_texts, show_progress_bar=False)
        return np.hstack((tfidf_features, sbert_embeddings * 1.5))

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['DNA_Target'] = df['Type'].apply(dna_mapping)

print("[+] Extracting TF-IDF + SBERT Features (Phase 1 with ngram_range=(1,3))...")
ext = Phase1ExperimentalExtractor(max_tfidf_features=50, ngram_range=(1,3))
X = ext.fit_transform(df['Requirement'].tolist())
y = df['DNA_Target'].values

algorithms = {
    "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n[+] Testing Phase 1 (N-Gram 1-3) on PROMISE (10-Fold CV)...")
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in algorithms.items():
    accs, f1s = [], []
    for tr, val in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[val])
        accs.append(accuracy_score(y[val], preds))
        f1s.append(f1_score(y[val], preds, average='macro', zero_division=0))
        
    print(f"{name:<25} | {np.mean(accs)*100:>10.2f}% | {np.mean(f1s)*100:>10.2f}%", flush=True)

print("="*60)
