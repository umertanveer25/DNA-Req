import pandas as pd
import numpy as np
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
import warnings
warnings.filterwarnings('ignore')

def dna_mapping_4class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'DROP')

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()

# Phase 0 Data (4-Class)
df_p0 = df.copy()
df_p0['Target'] = df_p0['Type'].apply(dna_mapping_4class)
df_p0 = df_p0[df_p0['Target'] != 'DROP']
y_p0 = df_p0['Target'].values

# Phase 1 & 1A Data (5-Class)
df_p1 = df.copy()
df_p1['Target'] = df_p1['Type'].apply(dna_mapping_5class)
y_p1 = df_p1['Target'].values

# Feature Extraction
print("Extracting Phase 0 Features (TF-IDF)...")
tfidf_p0 = TfidfVectorizer(max_features=3000, stop_words='english')
X_p0 = tfidf_p0.fit_transform(df_p0['Requirement']).toarray()

print("Extracting Phase 1 Features (TF-IDF)...")
tfidf_p1 = TfidfVectorizer(max_features=3000, stop_words='english')
X_p1 = tfidf_p1.fit_transform(df_p1['Requirement']).toarray()

print("Extracting Phase 1-A Features (TF-IDF + SBERT)...")
tfidf_1a = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1,3))
X_tfidf_1a = tfidf_1a.fit_transform(df_p1['Requirement']).toarray()
sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert_1a = sbert.encode(df_p1['Requirement'].tolist(), show_progress_bar=False)
X_p1a = np.hstack((X_tfidf_1a, X_sbert_1a * 1.5))

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

def evaluate(X, y):
    results = {}
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, clf in algorithms.items():
        accs = []
        try:
            X_cv = X
            if name == "Multinomial NB":
                scaler = MinMaxScaler()
                X_cv = scaler.fit_transform(X)
            for tr, val in skf.split(X_cv, y):
                clf.fit(X_cv[tr], y[tr])
                preds = clf.predict(X_cv[val])
                accs.append(accuracy_score(y[val], preds))
            results[name] = np.mean(accs) * 100
        except:
            results[name] = 0.0
    return results

print("Evaluating Phase 0...")
res_p0 = evaluate(X_p0, y_p0)
print("Evaluating Phase 1...")
res_p1 = evaluate(X_p1, y_p1)
print("Evaluating Phase 1-A...")
res_p1a = evaluate(X_p1a, y_p1)

print("\n" + "="*80)
print(f"{'Algorithm':<22} | {'Phase 0 (4-Class)':<17} | {'Phase 1 (5-Class)':<17} | {'Phase 1-A (5-Class)':<17}")
print("="*80)
for name in algorithms.keys():
    print(f"{name:<22} | {res_p0[name]:>16.2f}% | {res_p1[name]:>16.2f}% | {res_p1a[name]:>16.2f}%")
print("="*80)
