import os, sys, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from run_phase2_common import TextToDNAEncoder, dna_mapping_5class

# ---------------------------------------------------------------------------
# Data loading & encoding (Identical to Phase 2 Pipeline)
# ---------------------------------------------------------------------------
print("[+] Loading PROMISE dataset (Training)...")
df_train = pd.read_csv('data/Promise_Dataset.csv')
df_train['Type'] = df_train['Type'].str.strip()
df_train['Target'] = df_train['Type'].apply(dna_mapping_5class)
y_train = df_train['Target'].values
texts_train = df_train['Requirement'].tolist()

print("[+] Loading FNFC Dataset for Zero-Shot Evaluation (Testing)...")
df_test = pd.read_csv('C:/Users/umert/Downloads/FNFC.csv', encoding='ISO-8859-1')
text_col = 'Requirement' if 'Requirement' in df_test.columns else df_test.columns[0]
label_col = 'Type' if 'Type' in df_test.columns else ('Class' if 'Class' in df_test.columns else df_test.columns[1])
df_test[label_col] = df_test[label_col].astype(str).str.strip()
df_test['Target'] = df_test[label_col].apply(dna_mapping_5class)
y_test = df_test['Target'].values
texts_test = df_test[text_col].tolist()

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
print("[+] Extracting Phase 2 Features (DNA Codons + SBERT)...")
enc = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons_train = enc.fit_transform(texts_train, y_train.tolist())
X_codons_test = enc.transform(texts_test)

sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert_train = sbert.encode(texts_train, show_progress_bar=False)
X_sbert_test = sbert.encode(texts_test, show_progress_bar=False)

X_train_full = np.hstack((X_codons_train, X_sbert_train * 1.5))
X_test_full = np.hstack((X_codons_test, X_sbert_test * 1.5))
print(f"[+] Full Feature Space: Train {X_train_full.shape}, Test {X_test_full.shape}")

# ---------------------------------------------------------------------------
# Apply Phase 2A (GA) Optimizer Mask
# ---------------------------------------------------------------------------
print("[+] Applying Phase 2A (GA) Optimal Feature Mask...")
mask = np.load('features_opt_ga.npy')
X_train = X_train_full[:, mask]
X_test = X_test_full[:, mask]
print(f"[+] Optimized Feature Space: Train {X_train.shape}, Test {X_test.shape}")

# Scale for MultinomialNB
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

print(f"\n[+] Zero-Shot Testing 11 Algorithms on {len(df_test)} unseen FNFC requirements...")
print("="*65)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*65)

results = []
for name, clf in algorithms.items():
    X_tr = X_train_scaled if name == "Multinomial NB" else X_train
    X_te = X_test_scaled if name == "Multinomial NB" else X_test
    
    try:
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average='macro', zero_division=0) * 100
        print(f"{name:<25} | {acc:>10.2f}% | {f1:>10.2f}%", flush=True)
        results.append((name, acc, f1))
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*65)
df_res = pd.DataFrame(results, columns=['Algorithm', 'Accuracy', 'Macro F1'])
df_res.to_csv('results/phase2A_fnfc_zeroshot.csv', index=False)
print("[+] Results saved to results/phase2A_fnfc_zeroshot.csv")
