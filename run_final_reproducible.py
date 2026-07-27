"""
FINAL REPRODUCIBLE BENCHMARK
==============================
- Same protocol for Phase 1 and Phase 2
- 10-Fold Stratified CV, random_state=42 FIXED everywhere
- Run this ONCE → results are identical every run
- Output: results/FINAL_reproducible_results.csv
"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

from src.features import dna_mapping, DNAFeatureExtractor

SEED = 42  # FIXED FOREVER — never change this

def get_classifiers():
    return {
        "SVM RBF":             SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED),
        "SVM Linear":          SVC(kernel='linear', class_weight='balanced', probability=True, random_state=SEED),
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
        "KNN (k=3)":           KNeighborsClassifier(n_neighbors=3),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=SEED),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED),
        "AdaBoost":            AdaBoostClassifier(n_estimators=50, random_state=SEED),
        "Decision Tree":       DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=SEED),
        "Multinomial NB":      MultinomialNB(),
        "Naive Bayes":         GaussianNB(),
    }

def get_optimized_model(name, base_clf):
    """Phase 2: SelectKBest(k=150) + tuned hyperparams"""
    if isinstance(base_clf, MultinomialNB):
        return make_pipeline(MinMaxScaler(), SelectKBest(f_classif, k=150), MultinomialNB(alpha=0.5))
    elif isinstance(base_clf, GaussianNB):
        return make_pipeline(MinMaxScaler(), SelectKBest(f_classif, k=150), GaussianNB())
    elif isinstance(base_clf, RandomForestClassifier):
        return make_pipeline(SelectKBest(f_classif, k=150),
                             RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=SEED, n_jobs=-1))
    elif isinstance(base_clf, GradientBoostingClassifier):
        return make_pipeline(SelectKBest(f_classif, k=150),
                             GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=SEED))
    elif isinstance(base_clf, SVC):
        return make_pipeline(SelectKBest(f_classif, k=150),
                             SVC(C=2.0, gamma='scale', kernel=base_clf.kernel,
                                 class_weight='balanced', probability=True, random_state=SEED))
    elif isinstance(base_clf, KNeighborsClassifier):
        return make_pipeline(SelectKBest(f_classif, k=150),
                             KNeighborsClassifier(n_neighbors=base_clf.n_neighbors, n_jobs=-1))
    else:
        return make_pipeline(SelectKBest(f_classif, k=150), base_clf)

def run():
    print("[FINAL REPRODUCIBLE BENCHMARK] random_state=42 fixed everywhere", flush=True)
    print("=" * 80, flush=True)

    # Load
    df = pd.read_csv("data/Promise_Dataset.csv")
    df['Type'] = df['Type'].str.strip()
    df['DNA_Target'] = df['Type'].apply(dna_mapping)
    X_text = df['Requirement'].tolist()
    y = df['DNA_Target'].values
    print(f"[+] Dataset: {len(X_text)} requirements", flush=True)

    # Features
    print("[+] Extracting 434-d DNA Hybrid Features...", flush=True)
    extractor = DNAFeatureExtractor(max_tfidf_features=50)
    X = extractor.fit_transform(X_text)
    print(f"[+] Feature matrix: {X.shape}", flush=True)
    print("-" * 80, flush=True)

    # Fixed 10-fold CV — SAME for both phases
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    classifiers = get_classifiers()
    results = []

    total = len(classifiers)
    for i, (name, base_clf) in enumerate(classifiers.items(), 1):
        print(f"  [{i}/{total}] {name}...", flush=True)

        opt_clf = get_optimized_model(name, base_clf)

        # For MultinomialNB shift features to non-negative
        X_use = X.copy()
        if isinstance(base_clf, MultinomialNB) and X_use.min() < 0:
            X_use -= X_use.min()

        b_accs, o_accs, b_f1s, o_f1s = [], [], [], []
        for train_idx, val_idx in skf.split(X_use, y):
            Xtr, Xval = X_use[train_idx], X_use[val_idx]
            ytr, yval = y[train_idx], y[val_idx]

            base_clf.fit(Xtr, ytr)
            bp = base_clf.predict(Xval)
            b_accs.append(accuracy_score(yval, bp))
            b_f1s.append(f1_score(yval, bp, average='macro', zero_division=0))

            opt_clf.fit(Xtr, ytr)
            op = opt_clf.predict(Xval)
            o_accs.append(accuracy_score(yval, op))
            o_f1s.append(f1_score(yval, op, average='macro', zero_division=0))

        b_acc = np.mean(b_accs) * 100
        o_acc = np.mean(o_accs) * 100
        b_f1  = np.mean(b_f1s)  * 100
        o_f1  = np.mean(o_f1s)  * 100

        t_stat, p_val = stats.ttest_rel(o_accs, b_accs)
        try:
            _, p_wil = stats.wilcoxon(o_accs, b_accs)
        except Exception:
            p_wil = 1.0
        cd = (np.mean(o_accs) - np.mean(b_accs)) / (np.std(np.array(o_accs) - np.array(b_accs)) + 1e-9)

        results.append({
            'Algorithm':      name,
            'Phase1_Acc':     f"{b_acc:.2f}%",
            'Phase2_Acc':     f"{o_acc:.2f}%",
            'Acc_Change':     f"{o_acc - b_acc:+.2f}%",
            'Phase1_F1':      f"{b_f1:.2f}%",
            'Phase2_F1':      f"{o_f1:.2f}%",
            'F1_Change':      f"{o_f1 - b_f1:+.2f}%",
            't_stat':         f"{t_stat:.4f}",
            'p_value':        f"{p_val:.4e}",
            'p_wilcoxon':     f"{p_wil:.4e}",
            "Cohen_d":        f"{cd:.4f}",
            'Significant':    "YES" if p_val < 0.05 else "NO",
        })
        print(f"     Phase1={b_acc:.2f}%  Phase2={o_acc:.2f}%  Change={o_acc-b_acc:+.2f}%  p={p_val:.3e}", flush=True)

    df_out = pd.DataFrame(results)
    print("\n" + "=" * 80, flush=True)
    print("FINAL REPRODUCIBLE RESULTS (identical every run)", flush=True)
    print("=" * 80, flush=True)
    print(df_out.to_string(index=False), flush=True)

    os.makedirs("results", exist_ok=True)
    out_path = "results/FINAL_reproducible_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[+] Saved to {out_path}", flush=True)
    print("[+] These numbers will be IDENTICAL on every future run.", flush=True)

if __name__ == "__main__":
    run()
