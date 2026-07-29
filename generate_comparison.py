import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
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
import sys

sys.path.insert(0, '.')
from src.features import DNAEncoderTransformer

# 1. Load Data
df = pd.read_csv('data/Promise_Dataset.csv')
mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
df['Target'] = df['Type'].str.strip().apply(lambda x: mapping.get(str(x).upper(), 'N'))
y = df['Target'].values
texts = df['Requirement'].tolist()

# 2. Define 11 Algorithms
algorithms = {
    "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "SVM Linear": SVC(kernel='linear', C=10, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Multinomial NB": MultinomialNB(),
    "Naive Bayes": GaussianNB()
}

# 3. Generate Features for all phases
print("[+] Extracting TF-IDF (Phase 0)...")
X_p0 = TfidfVectorizer(max_features=1000).fit_transform(texts).toarray()
X_p0 = MinMaxScaler().fit_transform(X_p0)

print("[+] Extracting TF-IDF 98 + SBERT (Phase 1)...")
X_tfidf_98 = TfidfVectorizer(max_features=98, stop_words='english', ngram_range=(1,3)).fit_transform(texts).toarray()
sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert = sbert.encode(texts, show_progress_bar=False)
X_p1 = np.hstack((X_tfidf_98, X_sbert * 1.5))
X_p1_scaled = MinMaxScaler().fit_transform(X_p1)

print("[+] Extracting DNA 98 + SBERT (Phase 2)...")
dna_encoder = DNAEncoderTransformer(n_gram=3, max_features=98)
X_dna = dna_encoder.fit_transform(texts, y)
X_p2 = np.hstack((X_dna, X_sbert * 1.5))
X_p2_scaled = MinMaxScaler().fit_transform(X_p2)

# 4. Evaluate
def evaluate(X, X_scaled):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    for name, clf in algorithms.items():
        accs = []
        X_curr = X_scaled if name == "Multinomial NB" else X
        for tr, val in skf.split(X_curr, y):
            clf.fit(X_curr[tr], y[tr])
            preds = clf.predict(X_curr[val])
            accs.append(accuracy_score(y[val], preds))
        results[name] = np.mean(accs) * 100
    return results

print("[+] Evaluating Phase 0...")
res_p0 = evaluate(X_p0, X_p0)
print("[+] Evaluating Phase 1...")
res_p1 = evaluate(X_p1, X_p1_scaled)
print("[+] Evaluating Phase 2...")
res_p2 = evaluate(X_p2, X_p2_scaled)

# 5. Print Table
print("\n" + "="*80)
print(f"{'Algorithm':<25} | {'Phase 0 (TF-IDF)':<20} | {'Phase 1 (TF-IDF+SBERT)':<25} | {'Phase 2 (DNA+SBERT)':<20}")
print("="*80)
for algo in algorithms.keys():
    print(f"{algo:<25} | {res_p0[algo]:>18.2f}% | {res_p1[algo]:>23.2f}% | {res_p2[algo]:>18.2f}%")
print("="*80)
