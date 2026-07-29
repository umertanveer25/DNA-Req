import os, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from run_phase2_common import dna_mapping_5class

print("==================================================")
print(" PHASE 0: TRADITIONAL TF-IDF (ENGLISH) ZERO-SHOT  ")
print("==================================================")

# ---------------------------------------------------------------------------
# Load Train (PROMISE)
# ---------------------------------------------------------------------------
df_train = pd.read_csv('data/Promise_Dataset.csv')
df_train['Type'] = df_train['Type'].str.strip()
df_train['Target'] = df_train['Type'].apply(dna_mapping_5class)
y_train = df_train['Target'].values
texts_train = df_train['Requirement'].tolist()

# ---------------------------------------------------------------------------
# Load Test 1 (FNFC - 5 Class)
# ---------------------------------------------------------------------------
df_fnfc = pd.read_csv('C:/Users/umert/Downloads/FNFC.csv', encoding='ISO-8859-1')
text_col_fnfc = 'Requirement' if 'Requirement' in df_fnfc.columns else df_fnfc.columns[0]
label_col_fnfc = 'Type' if 'Type' in df_fnfc.columns else ('Class' if 'Class' in df_fnfc.columns else df_fnfc.columns[1])
df_fnfc[label_col_fnfc] = df_fnfc[label_col_fnfc].astype(str).str.strip()
df_fnfc['Target'] = df_fnfc[label_col_fnfc].apply(dna_mapping_5class)
y_fnfc = df_fnfc['Target'].values
texts_fnfc = df_fnfc[text_col_fnfc].tolist()

# ---------------------------------------------------------------------------
# Load Test 2 (FRNFR - Binary)
# ---------------------------------------------------------------------------
df_frnfr = pd.read_csv('C:/Users/umert/Downloads/reqs_frnfr_full.csv', encoding='ISO-8859-1')
df_frnfr = df_frnfr.dropna(subset=['domain'])
df_frnfr['domain'] = df_frnfr['domain'].astype(str).str.strip().str.upper()
y_frnfr_binary = np.array(['FR' if d == 'FR' else 'NFR' for d in df_frnfr['domain']])
texts_frnfr = df_frnfr['text'].tolist()

# ---------------------------------------------------------------------------
# Feature Extraction (Phase 0 = Standard TF-IDF on English)
# ---------------------------------------------------------------------------
# Match Phase 2 dimensions for fair comparison (482 features)
vectorizer = TfidfVectorizer(max_features=482, stop_words='english')
X_train = vectorizer.fit_transform(texts_train).toarray()
X_fnfc = vectorizer.transform(texts_fnfc).toarray()
X_frnfr = vectorizer.transform(texts_frnfr).toarray()

print(f"[+] Extracted Phase 0 Features: {X_train.shape[1]} dimensions")

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

results_fnfc = []
results_frnfr = []

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_fnfc_scaled = scaler.transform(X_fnfc)
X_frnfr_scaled = scaler.transform(X_frnfr)

print("\n[+] Training and Evaluating...")
for alg_name, clf in algorithms.items():
    X_tr = X_train_scaled if alg_name == "Multinomial NB" else X_train
    X_te_fnfc = X_fnfc_scaled if alg_name == "Multinomial NB" else X_fnfc
    X_te_frnfr = X_frnfr_scaled if alg_name == "Multinomial NB" else X_frnfr
    
    # Fit on PROMISE
    clf.fit(X_tr, y_train)
    
    # Predict FNFC
    preds_fnfc = clf.predict(X_te_fnfc)
    acc_fnfc = accuracy_score(y_fnfc, preds_fnfc) * 100
    
    # Predict FRNFR (Convert 5-class to Binary FR/NFR)
    preds_5class_frnfr = clf.predict(X_te_frnfr)
    preds_binary_frnfr = np.array(['FR' if p == 'A' else 'NFR' for p in preds_5class_frnfr])
    acc_frnfr = accuracy_score(y_frnfr_binary, preds_binary_frnfr) * 100
    
    results_fnfc.append((alg_name, acc_fnfc))
    results_frnfr.append((alg_name, acc_frnfr))

print("\n--- FNFC (5-Class) Zero-Shot Results ---")
for alg_name, acc in sorted(results_fnfc, key=lambda x: x[1], reverse=True):
    print(f"{alg_name:<25} | {acc:>6.2f}%")

print("\n--- FRNFR (Binary) Zero-Shot Results ---")
for alg_name, acc in sorted(results_frnfr, key=lambda x: x[1], reverse=True):
    print(f"{alg_name:<25} | {acc:>6.2f}%")

df_fnfc = pd.DataFrame(results_fnfc, columns=['Algorithm', 'Accuracy'])
df_frnfr = pd.DataFrame(results_frnfr, columns=['Algorithm', 'Accuracy'])
df_fnfc.to_csv('results/phase0_fnfc_zeroshot.csv', index=False)
df_frnfr.to_csv('results/phase0_frnfr_zeroshot.csv', index=False)
