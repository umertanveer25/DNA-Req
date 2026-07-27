import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['DNA_Target'] = df['Type'].apply(dna_mapping)

print("[+] Extracting TF-IDF Features (Phase 0 - No SBERT)...")
ext = DNAFeatureExtractor()
X = ext.fit_transform(df['Requirement'].tolist())
y = df['DNA_Target'].values

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
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42),
    "Multinomial NB": MultinomialNB(),
    "Naive Bayes": GaussianNB()
}

print("\n[+] Testing 12 Algorithms on PROMISE (10-Fold CV)...")
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in algorithms.items():
    accs, f1s = [], []
    try:
        for tr, val in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            preds = clf.predict(X[val])
            accs.append(accuracy_score(y[val], preds))
            f1s.append(f1_score(y[val], preds, average='macro', zero_division=0))
            
        print(f"{name:<25} | {np.mean(accs)*100:>10.2f}% | {np.mean(f1s)*100:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)
