import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from src.features import CanonicalDNAEncoder

def evaluate_fold_leakfree_precomputed_sbert(train_idx, val_idx, X_raw, X_sbert_global, y, classifier, selector=None):
    """
    Evaluates a single CV fold with isolated DNA feature extraction and scaling.
    Accepts pre-computed SBERT embeddings since frozen SBERT is unsupervised and does not leak labels.
    """
    X_train_raw = [X_raw[i] for i in train_idx]
    X_val_raw = [X_raw[i] for i in val_idx]
    
    X_train_sbert = X_sbert_global[train_idx]
    X_val_sbert = X_sbert_global[val_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]

    # 1. Fit DNA Encoder strictly on training fold (No Leakage)
    dna_enc = CanonicalDNAEncoder(step_size=3, sublinear_tf=True)
    dna_enc.fit(X_train_raw, y_train)
    X_tr_codons = dna_enc.transform(X_train_raw)
    X_val_codons = dna_enc.transform(X_val_raw)

    # 2. Hybrid Fusion with Pre-computed SBERT
    X_tr_fusion = np.hstack((X_tr_codons, X_train_sbert * 1.5))
    X_val_fusion = np.hstack((X_val_codons, X_val_sbert * 1.5))

    # 3. Fit MinMaxScaler strictly on training fold (No Leakage)
    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_fusion)
    X_val_scaled = scaler.transform(X_val_fusion)

    # 4. Fit Classifier
    classifier.fit(X_tr_scaled, y_train)
    preds = classifier.predict(X_val_scaled)
    
    return accuracy_score(y_val, preds)

def run_leak_free_cv_fast(X_raw, X_sbert_global, y, splits, classifier, selector=None):
    all_accuracies = []
    for split_idx, fold_pairs in enumerate(splits):
        fold_accs = []
        for train_idx, val_idx in fold_pairs:
            acc = evaluate_fold_leakfree_precomputed_sbert(
                train_idx, val_idx, X_raw, X_sbert_global, y, classifier, selector
            )
            fold_accs.append(acc)
        all_accuracies.append(np.mean(fold_accs) * 100)
    return np.mean(all_accuracies)
