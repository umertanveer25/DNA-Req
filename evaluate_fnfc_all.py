import os, sys, argparse, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Dataset Evaluation on FNFC (12 Algorithms)")
    parser.add_argument("--promise_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE training dataset")
    parser.add_argument("--fnfc_path", type=str, default="data/FNFC.csv", help="Path to FNFC test dataset")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.promise_path):
        raise FileNotFoundError(f"PROMISE dataset not found at {args.promise_path}")
        
    fnfc_candidates = [args.fnfc_path, "data/FNFC.csv", "C:/Users/umert/Downloads/FNFC.csv"]
    actual_fnfc_path = next((p for p in fnfc_candidates if os.path.exists(p)), None)
    if not actual_fnfc_path:
        raise FileNotFoundError(f"FNFC dataset not found at {args.fnfc_path} or default fallbacks.")

    print(f"[+] Loading PROMISE dataset: {args.promise_path}")
    df_train = pd.read_csv(args.promise_path)
    df_train['Type'] = df_train['Type'].str.strip()
    df_train['DNA_Target'] = df_train['Type'].apply(dna_mapping)
    y_train = df_train['DNA_Target'].values

    print(f"[+] Loading FNFC Dataset for Zero-Shot Evaluation: {actual_fnfc_path}")
    df_test = pd.read_csv(actual_fnfc_path, encoding='ISO-8859-1')
    text_col = 'Requirement' if 'Requirement' in df_test.columns else df_test.columns[0]
    label_col = 'Type' if 'Type' in df_test.columns else ('Class' if 'Class' in df_test.columns else df_test.columns[1])
    df_test[label_col] = df_test[label_col].astype(str).str.strip()
    df_test['DNA_Target'] = df_test[label_col].apply(dna_mapping)
    y_test = df_test['DNA_Target'].values

    print("[+] Extracting DNA Features (TF-IDF + SBERT)...")
    ext = DNAFeatureExtractor(max_tfidf_features=50)
    X_train = ext.fit_transform(df_train['Requirement'].tolist())
    X_test = ext.transform(df_test[text_col].tolist())

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
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42),
        "Multinomial NB": MultinomialNB(),
        "Naive Bayes": GaussianNB()
    }

    print(f"\n[+] Testing 12 Algorithms on {len(df_test)} unseen FNFC requirements...")
    print("=" * 60)
    print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
    print("=" * 60)

    for name, clf in algorithms.items():
        try:
            if name == "Multinomial NB":
                scaler = MinMaxScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = np.clip(scaler.transform(X_test), 0.0, 1.0)
            else:
                X_tr = X_train
                X_te = X_test

            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_test, preds) * 100
            f1 = f1_score(y_test, preds, average='macro', zero_division=0) * 100
            print(f"{name:<25} | {acc:>10.2f}% | {f1:>10.2f}%", flush=True)
        except Exception as e:
            print(f"{name:<25} | Error: {str(e)[:40]}")

    print("=" * 60)

if __name__ == "__main__":
    main()
