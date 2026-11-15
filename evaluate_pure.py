import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append(os.path.abspath('src'))
from features import normalize_promise_label, DNALinguisticTaxonomyExtractor
from models import get_paper_classifiers

def run_pure_experiment():
    print("=" * 60)
    print("🚀 RUNNING LIVE ZERO-SHOT GENERALIZABILITY EXPERIMENT: PROMISE -> PURE")
    print("=" * 60)

    # 1. Load PROMISE (Train)
    promise_path = os.path.join(os.path.dirname(__file__), 'data', 'Promise_Dataset.csv')
    df_promise = pd.read_csv(promise_path)
    
    # Text and labels
    train_texts = df_promise['Requirement'].astype(str).tolist()
    # Binary mapping: 'F' -> 0 (Functional), Non-'F' -> 1 (Non-Functional)
    raw_labels = df_promise['Type'].astype(str).str.strip().apply(normalize_promise_label)
    y_train = np.array([0 if lbl == 'F' else 1 for lbl in raw_labels])

    print(f"📦 Loaded PROMISE (Train): {len(train_texts)} samples")
    print(f"   Distribution: F (0) = {np.sum(y_train == 0)}, NFR (1) = {np.sum(y_train == 1)}")

    # 2. Load PURE (Zero-Shot Test)
    pure_path = os.path.join(os.path.dirname(__file__), 'data', 'PURE.csv')
    df_pure = pd.read_csv(pure_path, encoding='latin1')
    test_texts = df_pure['sentence'].astype(str).tolist()
    y_test = df_pure['NFR_boolean'].astype(int).values

    print(f"📦 Loaded PURE (Test): {len(test_texts)} samples")
    print(f"   Distribution: F (0) = {np.sum(y_test == 0)}, NFR (1) = {np.sum(y_test == 1)}")

    # 3. Feature Extraction
    print("\n⏳ Extracting TF-IDF Features...")
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()

    print("⏳ Extracting SBERT Semantic Embeddings (all-MiniLM-L6-v2)...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_sbert = sbert.encode(train_texts, show_progress_bar=False)
    X_test_sbert = sbert.encode(test_texts, show_progress_bar=True)

    print("⏳ Extracting Taxonomic DNA Features (28-D)...")
    dna_ext = DNALinguisticTaxonomyExtractor()
    X_train_dna = dna_ext.fit_transform(train_texts)
    X_test_dna = dna_ext.transform(test_texts)

    # Combined 462-D space
    X_train_462 = np.hstack([X_train_sbert, X_train_tfidf, X_train_dna])
    X_test_462 = np.hstack([X_test_sbert, X_test_tfidf, X_test_dna])

    print(f"✅ Features ready:")
    print(f"   TF-IDF Shape: Train {X_train_tfidf.shape}, Test {X_test_tfidf.shape}")
    print(f"   SBERT Shape:  Train {X_train_sbert.shape}, Test {X_test_sbert.shape}")
    print(f"   Hybrid 462-D: Train {X_train_462.shape}, Test {X_test_462.shape}")

    # Scale dense features
    scaler = StandardScaler()
    X_train_462_scaled = scaler.fit_transform(X_train_462)
    X_test_462_scaled = scaler.transform(X_test_462)

    # 4. Evaluate Classifiers
    # We compare TF-IDF vs SBERT/Hybrid representations
    classifiers = get_paper_classifiers(include_proposed_sequence_model=False)

    results = []

    print("\n" + "=" * 80)
    print(f"{'Classifier':<22} | {'TF-IDF Acc (%)':<15} | {'SBERT Acc (%)':<15} | {'Gain (%)':<10} | {'SBERT F1':<10}")
    print("=" * 80)

    for name, clf in classifiers.items():
        # Evaluate on TF-IDF
        clf_tfidf = type(clf)(**clf.get_params())
        clf_tfidf.fit(X_train_tfidf, y_train)
        pred_tfidf = clf_tfidf.predict(X_test_tfidf)
        acc_tfidf = accuracy_score(y_test, pred_tfidf) * 100

        # Evaluate on SBERT / Hybrid
        clf_sbert = type(clf)(**clf.get_params())
        if 'SVM' in name or 'Logistic' in name or 'KNN' in name:
            clf_sbert.fit(X_train_462_scaled, y_train)
            pred_sbert = clf_sbert.predict(X_test_462_scaled)
        else:
            clf_sbert.fit(X_train_462, y_train)
            pred_sbert = clf_sbert.predict(X_test_462)
        
        acc_sbert = accuracy_score(y_test, pred_sbert) * 100
        f1_sbert = f1_score(y_test, pred_sbert, average='macro')
        mcc_sbert = matthews_corrcoef(y_test, pred_sbert)
        kappa_sbert = cohen_kappa_score(y_test, pred_sbert)

        gain = acc_sbert - acc_tfidf

        results.append({
            'Classifier': name,
            'TFIDF_Acc': acc_tfidf,
            'SBERT_Acc': acc_sbert,
            'Gain': gain,
            'F1_Macro': f1_sbert,
            'MCC': mcc_sbert,
            'Kappa': kappa_sbert
        })

        print(f"{name:<22} | {acc_tfidf:<15.2f} | {acc_sbert:<15.2f} | {gain:<+10.2f} | {f1_sbert:<10.4f}")

    df_res = pd.DataFrame(results)
    print("=" * 80)
    print(f"{'Mean':<22} | {df_res['TFIDF_Acc'].mean():<15.2f} | {df_res['SBERT_Acc'].mean():<15.2f} | {df_res['Gain'].mean():<+10.2f} | {df_res['F1_Macro'].mean():<10.4f}")
    print("=" * 80)

    df_res.to_csv(os.path.join('data', 'pure_execution_results.csv'), index=False)
    print("\n✅ Results saved to data/pure_execution_results.csv")

if __name__ == '__main__':
    run_pure_experiment()
