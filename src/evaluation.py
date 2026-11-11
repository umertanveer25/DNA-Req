import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class BenchmarkEvaluator:
    """
    Implements Repeated 10-Fold Stratified Cross Validation with 30 randomized repeats
    (3,600 model evaluations in total across 12 classifiers) to compute 
    stability, generalization, and imbalance-aware metrics.
    Supports strict fold-pure feature extraction with zero data leakage,
    imbalance-aware metrics (Macro F1, Precision, Recall), and clean estimator cloning.
    """
    def __init__(self, classifiers: dict, random_state=42):
        self.classifiers = classifiers
        self.random_state = random_state

    def run_benchmark(self, X: np.ndarray = None, y: np.ndarray = None, num_splits=30,
                      texts: list = None, sbert_embeddings: np.ndarray = None, 
                      extractor = None) -> pd.DataFrame:
        """
        Executes the repeated stratified cross-validation loops (Repeated 10-Fold CV).
        Computes Accuracy, Macro F1, Macro Precision, and Macro Recall for every fold.
        """
        results_acc = {name: [] for name in self.classifiers.keys()}
        results_f1 = {name: [] for name in self.classifiers.keys()}
        results_prec = {name: [] for name in self.classifiers.keys()}
        results_rec = {name: [] for name in self.classifiers.keys()}
        
        # Determine split index source
        if sbert_embeddings is not None:
            split_source = sbert_embeddings
        elif X is not None:
            split_source = X
        else:
            raise ValueError("Either X or (texts, sbert_embeddings, extractor) must be provided.")

        print(f"🧬 Starting Repeated 10-Fold Stratified CV with {num_splits} independent repeats (Total splits: {num_splits * 10})...")
        if texts is not None and extractor is not None:
            print("🛡️ Fold-Pure Feature Isolation: ACTIVE (Zero test-set leakage in TF-IDF)")
        print("📊 Imbalance Metrics: Tracking Accuracy, Macro F1, Precision, and Recall")
        
        # Outer loop: 30 independent repeats
        for split_idx in range(num_splits):
            skf_split = StratifiedKFold(
                n_splits=10, 
                shuffle=True, 
                random_state=self.random_state + split_idx
            )
            
            for train_idx, test_idx in skf_split.split(split_source, y):
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 1. Feature Extraction (Fold-Pure vs Pre-extracted)
                if texts is not None and extractor is not None and sbert_embeddings is not None:
                    train_texts = [texts[i] for i in train_idx]
                    test_texts = [texts[i] for i in test_idx]
                    X_tr, X_te = extractor.fit_transform_fold(
                        train_texts, test_texts, 
                        sbert_embeddings[train_idx], sbert_embeddings[test_idx]
                    )
                else:
                    X_tr = X[train_idx]
                    X_te = X[test_idx]
                
                # 2. Benchmark each classifier
                for name, clf in self.classifiers.items():
                    # Multinomial NB mathematically requires non-negative inputs.
                    # Fit a fold-level MinMaxScaler to map features into [0, 1] without data snooping.
                    if name == "Multinomial NB":
                        nb_scaler = MinMaxScaler()
                        X_tr_clf = nb_scaler.fit_transform(X_tr)
                        X_te_clf = np.clip(nb_scaler.transform(X_te), 0.0, 1.0)
                    else:
                        X_tr_clf = X_tr
                        X_te_clf = X_te
                    
                    # Fresh model clone per fold (prevents state leakage)
                    fold_clf = clone(clf)
                    fold_clf.fit(X_tr_clf, y_train)
                    preds = fold_clf.predict(X_te_clf)
                    
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                    prec = precision_score(y_test, preds, average='macro', zero_division=0)
                    rec = recall_score(y_test, preds, average='macro', zero_division=0)
                    
                    results_acc[name].append(acc)
                    results_f1[name].append(f1)
                    results_prec[name].append(prec)
                    results_rec[name].append(rec)
                    
        # Compute summary statistics
        summary = []
        for name in self.classifiers.keys():
            acc_arr = np.array(results_acc[name]) * 100
            f1_arr = np.array(results_f1[name]) * 100
            prec_arr = np.array(results_prec[name]) * 100
            rec_arr = np.array(results_rec[name]) * 100
            
            summary.append({
                "Algorithm": name,
                "Mean Accuracy (%)": round(acc_arr.mean(), 2),
                "Acc Std (%)": round(acc_arr.std(), 2),
                "Mean Macro F1 (%)": round(f1_arr.mean(), 2),
                "F1 Std (%)": round(f1_arr.std(), 2),
                "Mean Precision (%)": round(prec_arr.mean(), 2),
                "Mean Recall (%)": round(rec_arr.mean(), 2)
            })
            
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values(by="Mean Macro F1 (%)", ascending=False)
        return summary_df
