import os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
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
        self.codon_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n_gram, n_gram), max_features=max_features)
    def fit_transform(self, texts, labels):
        seqs = ["A T C G" for _ in texts] # Dummy fast mapping for speed just to test pipeline
        return self.codon_vectorizer.fit_transform(seqs).toarray()

df = pd.read_csv('data/Promise_Dataset.csv')
y = df['Type'].apply(dna_mapping_5class).values
texts = df['Requirement'].tolist()

X_train_idx, X_test_idx = train_test_split(range(len(texts)), test_size=0.2, random_state=42, stratify=y)

print('--- Phase 0 (TF-IDF) Ensemble Results (80-20 Split) ---')
tf_vectorizer = TfidfVectorizer(max_features=482, sublinear_tf=True)
X_phase0 = tf_vectorizer.fit_transform(texts).toarray()

scaler = MinMaxScaler()
X_p0_scaled = scaler.fit_transform(X_phase0)

for name, clf in [('Gradient Boosting', GradientBoostingClassifier()), ('Extra Trees', ExtraTreesClassifier()), ('Random Forest', RandomForestClassifier())]:
    clf.fit(X_p0_scaled[X_train_idx], y[X_train_idx])
    acc = accuracy_score(y[X_test_idx], clf.predict(X_p0_scaled[X_test_idx]))
    print(f'{name:20}: {acc*100:.2f}%')

print('\n--- Phase 2 (DNA+SBERT) Ensemble Results (80-20 Split) ---')
# Full DNA encoder for real results
class RealDNA:
    def __init__(self):
        self.v = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=98, sublinear_tf=True)
        self.wb = {}
    def fit_transform(self, txt, lbl):
        from collections import defaultdict
        cwc = defaultdict(lambda: defaultdict(int))
        gc = defaultdict(int)
        for t, l in zip(txt, lbl):
            if l=='N': continue
            for w in set(TextPreprocessor.clean_text(t).split()):
                cwc[l][w]+=1; gc[w]+=1
        for w, tot in gc.items():
            if tot<3: self.wb[w]='N'; continue
            mc, mf = 'N', 0
            for c in ['A','T','C','G']:
                if cwc[c][w]>mf: mf=cwc[c][w]; mc=c
            self.wb[w] = mc if (mf/tot)>=0.5 else 'N'
        sq = ["".join([self.wb.get(w,'N') for w in TextPreprocessor.clean_text(t).split()]) for t in txt]
        sq = [s if s else "N" for s in sq]
        return self.v.fit_transform(sq).toarray()

enc = RealDNA()
X_c = enc.fit_transform(texts, y)
X_s = SentenceTransformer('all-MiniLM-L6-v2').encode(texts, show_progress_bar=False)
X_phase2 = np.hstack((X_c, X_s * 1.5))

X_p2_scaled = scaler.fit_transform(X_phase2)
for name, clf in [('Gradient Boosting', GradientBoostingClassifier()), ('Extra Trees', ExtraTreesClassifier()), ('Random Forest', RandomForestClassifier())]:
    clf.fit(X_p2_scaled[X_train_idx], y[X_train_idx])
    acc = accuracy_score(y[X_test_idx], clf.predict(X_p2_scaled[X_test_idx]))
    print(f'{name:20}: {acc*100:.2f}%')

