import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath('src'))
from features import normalize_promise_label, DNALinguisticTaxonomyExtractor
from models import get_paper_classifiers

# PROMISE 12 canonical classes
PROMISE_CLASSES = ['F', 'SE', 'US', 'O', 'PE', 'LF', 'A', 'MN', 'SC', 'FT', 'L', 'PO']
PROMISE_LABEL_TO_IDX = {c: i for i, c in enumerate(PROMISE_CLASSES)}

# FNFC 14 classes
FNFC_CLASSES = ['F', 'SE', 'AU', 'P', 'A', 'PE', 'US', 'O', 'LL', 'R', 'M', 'LF', 'FT', 'SC']

def map_promise_pred_to_fnfc(pred_promise_class):
    """
    Taxonomic mapping from PROMISE prediction to FNFC class space:
    L -> LL, PO -> P, MN -> M, R -> {A, FT}
    """
    mapping = {
        'L': 'LL',
        'PO': 'P',
        'MN': 'M'
    }
    return mapping.get(pred_promise_class, pred_promise_class)

def evaluate_multi_class_generalization():
    print("=" * 80)
    print("🚀 RUNNING MULTI-CLASS ZERO-SHOT GENERALIZATION: PROMISE (12-Class) -> FNFC (14-Class)")
    print("=" * 80)

    # 1. Load PROMISE (12-Class Multi-Class)
    promise_path = os.path.join(os.path.dirname(__file__), 'data', 'Promise_Dataset.csv')
    df_promise = pd.read_csv(promise_path)
    train_texts = df_promise['Requirement'].astype(str).tolist()
    y_train_raw = df_promise['Type'].astype(str).str.strip().apply(normalize_promise_label).values
    
    # Filter only valid PROMISE classes
    y_train = y_train_raw
    print(f"📦 Loaded PROMISE Training Set: {len(train_texts)} requirements")
    print(f"   Classes (12 Multi-Class): {pd.Series(y_train).value_counts().to_dict()}")

    # 2. Load FNFC (14-Class Multi-Class)
    fnfc_path = os.path.join(os.path.dirname(__file__), 'data', 'FNFC.csv')
    df_fnfc = pd.read_csv(fnfc_path, encoding='ISO-8859-1')
    test_texts = df_fnfc['text'].astype(str).tolist()
    y_test_fnfc = df_fnfc['class'].astype(str).str.strip().values
    print(f"\n📦 Loaded FNFC Target Set: {len(test_texts)} requirements")
    print(f"   Classes (14 Multi-Class): {pd.Series(y_test_fnfc).value_counts().to_dict()}")

    # 3. Extract Features
    print("\n⏳ Extracting TF-IDF Features (50-D)...")
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()

    print("⏳ Extracting SBERT Semantic Embeddings (384-D)...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_sbert = sbert.encode(train_texts, show_progress_bar=False)
    X_test_sbert = sbert.encode(test_texts, show_progress_bar=True)

    print("⏳ Extracting DNA-Linguistic Features (28-D)...")
    dna_ext = DNALinguisticTaxonomyExtractor()
    X_train_dna = dna_ext.extract_batch(train_texts)
    X_test_dna = dna_ext.extract_batch(test_texts)

    # 462-D Hybrid Representation
    X_train_462 = np.hstack([X_train_sbert, X_train_tfidf, X_train_dna])
    X_test_462 = np.hstack([X_test_sbert, X_test_tfidf, X_test_dna])

    # Scaler for continuous feature spaces
    scaler = StandardScaler()
    X_train_462_scaled = scaler.fit_transform(X_train_462)
    X_test_462_scaled = scaler.transform(X_test_462)

    classifiers = get_paper_classifiers(include_proposed_sequence_model=False)

    results = []

    print("\n" + "=" * 95)
    print(f"{'Classifier':<20} | {'TF-IDF Acc':<12} | {'SBERT Acc':<12} | {'Gain':<10} | {'SBERT F1':<10} | {'MCC':<8} | {'Kappa':<8}")
    print("=" * 95)

    for name, clf in classifiers.items():
        # A. Evaluate TF-IDF Baseline
        clf_tfidf = type(clf)(**clf.get_params())
        if name == "Multinomial NB":
            mm_scaler = MinMaxScaler()
            X_tr_tfidf = mm_scaler.fit_transform(X_train_tfidf)
            X_te_tfidf = mm_scaler.transform(X_test_tfidf)
            clf_tfidf.fit(X_tr_tfidf, y_train)
            raw_preds_tfidf = clf_tfidf.predict(X_te_tfidf)
        else:
            clf_tfidf.fit(X_train_tfidf, y_train)
            raw_preds_tfidf = clf_tfidf.predict(X_test_tfidf)

        preds_tfidf_fnfc = [map_promise_pred_to_fnfc(p) for p in raw_preds_tfidf]
        acc_tfidf = accuracy_score(y_test_fnfc, preds_tfidf_fnfc) * 100
        f1_tfidf = f1_score(y_test_fnfc, preds_tfidf_fnfc, average='macro', zero_division=0)

        # B. Evaluate SBERT + DNA 462-D Hybrid
        clf_sbert = type(clf)(**clf.get_params())
        if 'SVM' in name or 'Logistic' in name or 'KNN' in name:
            clf_sbert.fit(X_train_462_scaled, y_train)
            raw_preds_sbert = clf_sbert.predict(X_test_462_scaled)
        elif name == "Multinomial NB":
            mm_scaler = MinMaxScaler()
            X_tr_mm = mm_scaler.fit_transform(X_train_462)
            X_te_mm = np.clip(mm_scaler.transform(X_test_462), 0.0, 1.0)
            clf_sbert.fit(X_tr_mm, y_train)
            raw_preds_sbert = clf_sbert.predict(X_te_mm)
        else:
            clf_sbert.fit(X_train_462, y_train)
            raw_preds_sbert = clf_sbert.predict(X_test_462)

        preds_sbert_fnfc = [map_promise_pred_to_fnfc(p) for p in raw_preds_sbert]
        acc_sbert = accuracy_score(y_test_fnfc, preds_sbert_fnfc) * 100
        f1_sbert = f1_score(y_test_fnfc, preds_sbert_fnfc, average='macro', zero_division=0)
        mcc_sbert = matthews_corrcoef(y_test_fnfc, preds_sbert_fnfc)
        kappa_sbert = cohen_kappa_score(y_test_fnfc, preds_sbert_fnfc)

        gain = acc_sbert - acc_tfidf

        results.append({
            'Classifier': name,
            'TFIDF_Acc': acc_tfidf,
            'SBERT_Acc': acc_sbert,
            'Gain': gain,
            'TFIDF_F1': f1_tfidf,
            'SBERT_F1': f1_sbert,
            'MCC': mcc_sbert,
            'Kappa': kappa_sbert
        })

        print(f"{name:<20} | {acc_tfidf:>10.2f}% | {acc_sbert:>10.2f}% | {gain:>+9.2f}% | {f1_sbert:>10.4f} | {mcc_sbert:>8.3f} | {kappa_sbert:>8.3f}")

    df_res = pd.DataFrame(results)
    print("=" * 95)
    print(f"{'Mean':<20} | {df_res['TFIDF_Acc'].mean():>10.2f}% | {df_res['SBERT_Acc'].mean():>10.2f}% | {df_res['Gain'].mean():>+9.2f}% | {df_res['SBERT_F1'].mean():>10.4f} | {df_res['MCC'].mean():>8.3f} | {df_res['Kappa'].mean():>8.3f}")
    print("=" * 95)

    df_res.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'fnfc_multiclass_live_results.csv'), index=False)
    print("\n✅ Multi-class generalization results saved to data/fnfc_multiclass_live_results.csv")

if __name__ == '__main__':
    evaluate_multi_class_generalization()
