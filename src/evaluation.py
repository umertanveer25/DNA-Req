import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class BenchmarkEvaluator:
    """
    Implements 10-Fold Stratified Cross Validation with 30 randomized splits 
    (3,600 model evaluations in total across 12 classifiers) to compute 
    stability and generalization metrics.
    """
    def __init__(self, classifiers: dict, random_state=42):
        self.classifiers = classifiers
        self.random_state = random_state

    def run_benchmark(self, X: np.ndarray, y: np.ndarray, num_splits=30) -> pd.DataFrame:
        """
        Executes the cross-validation loops and returns a DataFrame of results.
        """
        results = {name: [] for name in self.classifiers.keys()}
        
        # Ensure we can run Multinomial NB by shifting features to non-negative values
        # since SBERT features can be negative
        X_non_negative = X.copy()
        if X_non_negative.min() < 0:
            X_non_negative -= X_non_negative.min()

        # Outer 10-fold cross validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        print(f"🧬 Starting 10-Fold CV with {num_splits} randomized splits per fold...")
        
        # To simulate the 30 randomized splits per fold (30 iterations of cross-validation)
        for split_idx in range(num_splits):
            # Re-initialize or use a different seed/shuffle for each split run to get stable averages
            skf_split = StratifiedKFold(
                n_splits=10, 
                shuffle=True, 
                random_state=self.random_state + split_idx
            )
            
            for train_idx, test_idx in skf_split.split(X, y):
                y_train, y_test = y[train_idx], y[test_idx]
                
                for name, clf in self.classifiers.items():
                    # Select non-negative feature space for Multinomial NB specifically
                    X_tr = X_non_negative[train_idx] if name == "Multinomial NB" else X[train_idx]
                    X_te = X_non_negative[test_idx] if name == "Multinomial NB" else X[test_idx]
                    
                    # Fit model
                    clf.fit(X_tr, y_train)
                    preds = clf.predict(X_te)
                    
                    acc = accuracy_score(y_test, preds)
                    results[name].append(acc)
                    
        # Compute summary statistics
        summary = []
        for name, scores in results.items():
            scores_arr = np.array(scores)
            summary.append({
                "Algorithm": name,
                "Mean Accuracy (%)": round(scores_arr.mean() * 100, 2),
                "Std Dev (%)": round(scores_arr.std() * 100, 2),
                "Min (%)": round(scores_arr.min() * 100, 2),
                "Max (%)": round(scores_arr.max() * 100, 2),
                "Median (%)": round(np.median(scores_arr) * 100, 2)
            })
            
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values(by="Mean Accuracy (%)", ascending=False)
        return summary_df
