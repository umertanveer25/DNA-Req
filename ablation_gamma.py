import os, sys, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Gamma Parameter Sensitivity and Ablation Study")
    parser.add_argument("--data_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE dataset")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save ablation results")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save ablation plot")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print("🧬 Starting Gamma (γ) Semantic Scaling Sensitivity Analysis...")
    print("=" * 65)

    df = pd.read_csv(args.data_path)
    df['Type'] = df['Type'].str.strip()
    df['DNA_Target'] = df['Type'].apply(dna_mapping)

    texts = df['Requirement'].tolist()
    y = df['DNA_Target'].values

    # Pre-compute SBERT once
    extractor = DNAFeatureExtractor()
    sbert_emb = extractor.encode_sbert(texts)

    gammas = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    models = {
        "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        "SVM Linear": SVC(kernel='linear', C=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    records = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for g in gammas:
        print(f"[+] Evaluating gamma = {g:.2f}...")
        extractor.gamma = g

        gamma_accs = {m: [] for m in models}
        gamma_f1s = {m: [] for m in models}

        for train_idx, test_idx in skf.split(sbert_emb, y):
            tr_texts = [texts[i] for i in train_idx]
            te_texts = [texts[i] for i in test_idx]
            
            X_tr, X_te = extractor.fit_transform_fold(
                tr_texts, te_texts, sbert_emb[train_idx], sbert_emb[test_idx]
            )
            y_tr, y_te = y[train_idx], y[test_idx]

            for m_name, model in models.items():
                clf = clone(model)
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                
                gamma_accs[m_name].append(accuracy_score(y_te, preds))
                gamma_f1s[m_name].append(f1_score(y_te, preds, average='macro', zero_division=0))

        for m_name in models:
            records.append({
                "Gamma": g,
                "Model": m_name,
                "Mean Accuracy (%)": round(np.mean(gamma_accs[m_name]) * 100, 2),
                "Acc Std (%)": round(np.std(gamma_accs[m_name]) * 100, 2),
                "Mean Macro F1 (%)": round(np.mean(gamma_f1s[m_name]) * 100, 2),
                "F1 Std (%)": round(np.std(gamma_f1s[m_name]) * 100, 2)
            })

    res_df = pd.DataFrame(records)
    csv_path = os.path.join(args.out_dir, "gamma_ablation_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved ablation table to: {csv_path}")
    print(res_df.to_string(index=False))

    # Plotting Sensitivity Curves
    plt.figure(figsize=(10, 6))
    for m_name in models:
        sub = res_df[res_df['Model'] == m_name]
        plt.plot(sub['Gamma'], sub['Mean Accuracy (%)'], marker='o', lw=2, label=f"{m_name} Accuracy")
        plt.plot(sub['Gamma'], sub['Mean Macro F1 (%)'], marker='s', lw=1.5, linestyle='--', label=f"{m_name} Macro F1")

    plt.axvline(x=1.5, color='red', linestyle=':', lw=1.5, label='Selected Golden Factor (γ = 1.5)')
    plt.title("Sensitivity Analysis of Semantic Scaling Factor (γ) on 10-Fold CV", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Semantic Scaling Factor (γ)", fontsize=11)
    plt.ylabel("Performance (%)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.plots_dir, "gamma_sensitivity_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 Saved Sensitivity Plot to: {plot_path}")

if __name__ == "__main__":
    main()
