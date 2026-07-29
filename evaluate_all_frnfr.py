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
# Data loading & encoding
# ---------------------------------------------------------------------------
print("[+] Loading PROMISE dataset (Training, 5-Class)...")
df_train = pd.read_csv('data/Promise_Dataset.csv')
df_train['Type'] = df_train['Type'].str.strip()
df_train['Target'] = df_train['Type'].apply(dna_mapping_5class)
y_train = df_train['Target'].values
texts_train = df_train['Requirement'].tolist()

print("[+] Loading FRNFR Dataset for Zero-Shot Evaluation (Testing, Binary)...")
df_test = pd.read_csv('C:/Users/umert/Downloads/reqs_frnfr_full.csv', encoding='ISO-8859-1')
# The FRNFR dataset has columns ['id', 'text', 'domain']
df_test = df_test.dropna(subset=['domain'])
df_test['domain'] = df_test['domain'].astype(str).str.strip().str.upper()

# True labels for evaluation
y_test_binary = np.array(['FR' if d == 'FR' else 'NFR' for d in df_test['domain']])
texts_test = df_test['text'].tolist()

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
print("[+] Extracting Phase Features (DNA Codons + SBERT)...")
enc = TextToDNAEncoder(n_gram=3, max_features=98)
X_codons_train = enc.fit_transform(texts_train, y_train.tolist())
X_codons_test = enc.transform(texts_test)

sbert = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert_train = sbert.encode(texts_train, show_progress_bar=False)
X_sbert_test = sbert.encode(texts_test, show_progress_bar=False)

X_phase2_train = np.hstack((X_codons_train, X_sbert_train * 1.5))
X_phase2_test = np.hstack((X_codons_test, X_sbert_test * 1.5))

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

phases = {
    "Phase 1 (Codons Only)": (X_codons_train, X_codons_test),
    "Phase 2 (Fusion Baseline)": (X_phase2_train, X_phase2_test),
    "Phase 2A (GA)": (X_phase2_train[:, np.load('features_opt_ga.npy')], X_phase2_test[:, np.load('features_opt_ga.npy')]),
    "Phase 2B (PSO)": (X_phase2_train[:, np.load('features_opt_pso.npy')], X_phase2_test[:, np.load('features_opt_pso.npy')]),
    "Phase 2C (ACO)": (X_phase2_train[:, np.load('features_opt_aco.npy')], X_phase2_test[:, np.load('features_opt_aco.npy')]),
    "Phase 2D (ABC)": (X_phase2_train[:, np.load('features_opt_abc.npy')], X_phase2_test[:, np.load('features_opt_abc.npy')]),
    "Phase 2E (GWO)": (X_phase2_train[:, np.load('features_opt_gwo.npy')], X_phase2_test[:, np.load('features_opt_gwo.npy')]),
    "Phase 2F (WOA)": (X_phase2_train[:, np.load('features_opt_woa.npy')], X_phase2_test[:, np.load('features_opt_woa.npy')]),
}

all_results = []

for phase_name, (X_tr_base, X_te_base) in phases.items():
    print(f"\n[================ {phase_name} ================]")
    print(f"Features: {X_tr_base.shape[1]} dims")
    
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_base)
    X_te_scaled = scaler.transform(X_te_base)

    for alg_name, clf in algorithms.items():
        X_tr = X_tr_scaled if alg_name == "Multinomial NB" else X_tr_base
        X_te = X_te_scaled if alg_name == "Multinomial NB" else X_te_base
        
        try:
            clf.fit(X_tr, y_train)
            preds_5class = clf.predict(X_te)
            
            # Convert 5-class DNA predictions to binary for FRNFR evaluation
            # 'A' was mapped from 'F' (Functional). Everything else (T, C, G, N) is NFR.
            preds_binary = np.array(['FR' if p == 'A' else 'NFR' for p in preds_5class])
            
            acc = accuracy_score(y_test_binary, preds_binary) * 100
            f1 = f1_score(y_test_binary, preds_binary, average='macro', zero_division=0) * 100
            
            print(f"{alg_name:<25} | {acc:>6.2f}% | {f1:>6.2f}%", flush=True)
            all_results.append((phase_name, alg_name, acc, f1))
        except Exception as e:
            print(f"{alg_name:<25} | Error: {str(e)[:40]}")

df_res = pd.DataFrame(all_results, columns=['Phase', 'Algorithm', 'Accuracy', 'Macro_F1'])
df_res.to_csv('results/all_phases_frnfr_zeroshot.csv', index=False)
print("\n[+] All Results saved to results/all_phases_frnfr_zeroshot.csv")
