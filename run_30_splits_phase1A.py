import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['Target'] = df['Type'].apply(dna_mapping_5class)
y = df['Target'].values

print("[+] Extracting TF-IDF (98 dimensions) + SBERT (384 dimensions) = 482 Dimensions...")
tfidf = TfidfVectorizer(max_features=98, stop_words='english', ngram_range=(1,3))
X_tfidf = tfidf.fit_transform(df['Requirement']).toarray()

sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert = sbert.encode(df['Requirement'].tolist(), show_progress_bar=False)

X = np.hstack((X_tfidf, X_sbert * 1.5))
print(f"[+] Final Feature Matrix Shape: {X.shape} (Matches exactly 482 dimensions!)")

algorithms = {
    "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale'),
    "SVM Linear": SVC(kernel='linear', C=10),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=50),
    "Decision Tree": DecisionTreeClassifier(),
    "Multinomial NB": MultinomialNB(),
    "Naive Bayes": GaussianNB()
}

print("\n[+] Running 10-Fold CV with 30 Randomized Splits (300 Folds per Algorithm)...")
num_splits = 30
results = {name: [] for name in algorithms.keys()}

# Pre-scale for MultinomialNB
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

import time
start_time = time.time()

for split_idx in range(num_splits):
    # Use split_idx as random_state to ensure 30 completely different randomized splits
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_idx)
    
    for name, clf in algorithms.items():
        try:
            X_cv = X_scaled if name == "Multinomial NB" else X
            split_accs = []
            for tr, val in skf.split(X_cv, y):
                clf.fit(X_cv[tr], y[tr])
                preds = clf.predict(X_cv[val])
                split_accs.append(accuracy_score(y[val], preds))
            
            # Average accuracy for this specific 10-fold split
            results[name].append(np.mean(split_accs) * 100)
        except Exception as e:
            results[name].append(0.0)
            
    if (split_idx + 1) % 5 == 0:
        print(f"    -> Completed {split_idx + 1}/{num_splits} Randomized Splits...")

print(f"\n[+] Completed 30 Randomized Splits in {time.time() - start_time:.2f} seconds!")
print("="*80)
print(f"{'Algorithm':<25} | {'10-Fold 30-Split Avg Accuracy':<30}")
print("="*80)

final_averages = {}
for name in algorithms.keys():
    avg_acc = np.mean(results[name])
    final_averages[name] = avg_acc

# Sort by accuracy descending
sorted_algs = sorted(final_averages.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_algs:
    print(f"{name:<25} | {acc:>25.2f}%")
print("="*80)
