import os, sys, argparse, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Dataset Evaluation on FNFC (SVM RBF)")
    parser.add_argument("--promise_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE training dataset")
    parser.add_argument("--fnfc_path", type=str, default="data/FNFC.csv", help="Path to FNFC test dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.promise_path):
        raise FileNotFoundError(f"PROMISE dataset not found at {args.promise_path}")
    
    # Check FNFC path with fallbacks
    fnfc_candidates = [args.fnfc_path, "data/FNFC.csv", "C:/Users/umert/Downloads/FNFC.csv"]
    actual_fnfc_path = next((p for p in fnfc_candidates if os.path.exists(p)), None)
    if not actual_fnfc_path:
        raise FileNotFoundError(f"FNFC dataset not found at {args.fnfc_path} or default fallbacks.")

    print(f"[+] Training SVM RBF on PROMISE dataset: {args.promise_path}")
    df_train = pd.read_csv(args.promise_path)
    df_train['Type'] = df_train['Type'].str.strip()
    df_train['DNA_Target'] = df_train['Type'].apply(dna_mapping)

    ext = DNAFeatureExtractor(max_tfidf_features=50)
    X_train = ext.fit_transform(df_train['Requirement'].tolist())
    y_train = df_train['DNA_Target'].values

    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)

    print(f"[+] Loading FNFC Dataset for Zero-Shot Evaluation: {actual_fnfc_path}")
    df_test = pd.read_csv(actual_fnfc_path, encoding='ISO-8859-1')
    
    text_col = 'Requirement' if 'Requirement' in df_test.columns else df_test.columns[0]
    label_col = 'Type' if 'Type' in df_test.columns else ('Class' if 'Class' in df_test.columns else df_test.columns[1])

    df_test[label_col] = df_test[label_col].astype(str).str.strip()
    df_test['DNA_Target'] = df_test[label_col].apply(dna_mapping)

    X_test = ext.transform(df_test[text_col].tolist())
    y_test = df_test['DNA_Target'].values

    print(f"[+] Testing on {len(df_test)} unseen FNFC requirements...")
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)

    print("=" * 50)
    print(f"FNFC Generalizability Accuracy: {acc * 100:.2f}%")
    print(f"FNFC Generalizability Macro F1: {f1 * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
