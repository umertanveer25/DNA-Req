import os
import sys
import argparse
import multiprocessing
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor
from src.models import get_paper_classifiers, get_proposed_sequence_model

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-Accelerated DNA-Req Benchmark Pipeline")
    parser.add_argument("--data_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE dataset")
    parser.add_argument("--splits", type=int, default=30, help="Number of randomized repeats (default: 30 for 3,600 runs)")
    parser.add_argument("--demo", action="store_true", help="Run 1 repeat (10 folds) for rapid testing")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save classification results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Device Diagnostics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("🚀 DNA-Req GPU-Accelerated Classification Pipeline")
    print("=" * 65)
    print(f"🖥️ Hardware Acceleration Engine: {device.upper()}")
    if device == "cuda":
        print(f"⚡ GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"⚡ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"⚡ PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ No CUDA GPU detected, running on multi-core CPU mode.")
    print("-" * 65)

    # 2. Load Dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")
        
    df = pd.read_csv(args.data_path)
    df['Type'] = df['Type'].str.strip()
    df['DNA_Target'] = df['Type'].apply(dna_mapping)
    
    X_text = df['Requirement'].tolist()
    y = df['DNA_Target'].values

    print(f"📊 Dataset loaded: {len(X_text)} requirements.")
    print("📋 Class Distribution:")
    for k, v in df['DNA_Target'].value_counts().items():
        print(f"   Base {k}: {v} samples ({v/len(df)*100:.1f}%)")
    print("-" * 65)

    # 3. GPU-Accelerated SBERT Extraction
    print(f"🧬 Extracting Contextual SBERT Representations on {device.upper()}...")
    extractor = DNAFeatureExtractor()
    extractor.sbert_model = extractor.sbert_model.to(device)
    
    # Batch encode on GPU
    cleaned_texts = [t.strip() for t in X_text]
    with torch.no_grad():
        sbert_embeddings = extractor.sbert_model.encode(
            cleaned_texts, 
            batch_size=128 if device == "cuda" else 32,
            device=device,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    print(f"✅ SBERT Embeddings: {sbert_embeddings.shape} (Normalized on {device.upper()})")
    print("-" * 65)

    # 4. Initialize Classifiers (12 Classical + Proposed Deep Sequence Model)
    classifiers = get_paper_classifiers(include_proposed_sequence_model=True)
    num_splits = 1 if args.demo else args.splits
    total_runs = num_splits * 10 * len(classifiers)

    print(f"🏋️ Model Benchmarking: {len(classifiers)} Algorithms")
    print(f"🔁 Repeated 10-Fold Stratified CV with {num_splits} independent repeats")
    print(f"🎯 Total Evaluations: {num_splits} repeats × 10 folds × {len(classifiers)} models = {total_runs:,} model runs")
    print("🛡️ Fold-Pure Feature Isolation: ACTIVE (Zero TF-IDF data leakage)")
    print("-" * 65)

    results_acc = {name: [] for name in classifiers.keys()}
    results_f1 = {name: [] for name in classifiers.keys()}
    results_prec = {name: [] for name in classifiers.keys()}
    results_rec = {name: [] for name in classifiers.keys()}

    # 5. Run Cross-Validation Loop
    pbar = tqdm(total=num_splits * 10, desc="⚡ CV Progress")
    for split_idx in range(num_splits):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + split_idx)
        
        for train_idx, test_idx in skf.split(sbert_embeddings, y):
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fold-pure feature extraction
            tr_texts = [X_text[i] for i in train_idx]
            te_texts = [X_text[i] for i in test_idx]
            X_tr, X_te = extractor.fit_transform_fold(
                tr_texts, te_texts, sbert_embeddings[train_idx], sbert_embeddings[test_idx]
            )
            
            # Fit each model
            for name, clf in classifiers.items():
                if name == "Multinomial NB":
                    scaler = MinMaxScaler()
                    X_tr_clf = scaler.fit_transform(X_tr)
                    X_te_clf = np.clip(scaler.transform(X_te), 0.0, 1.0)
                else:
                    X_tr_clf = X_tr
                    X_te_clf = X_te
                
                fold_clf = clone(clf)
                fold_clf.fit(X_tr_clf, y_train)
                preds = fold_clf.predict(X_te_clf)
                
                results_acc[name].append(accuracy_score(y_test, preds))
                results_f1[name].append(f1_score(y_test, preds, average='macro', zero_division=0))
                results_prec[name].append(precision_score(y_test, preds, average='macro', zero_division=0))
                results_rec[name].append(recall_score(y_test, preds, average='macro', zero_division=0))
            
            pbar.update(1)

    pbar.close()

    # 6. Compile and Format Results
    summary = []
    for name in classifiers.keys():
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
            "Precision (%)": round(prec_arr.mean(), 2),
            "Recall (%)": round(rec_arr.mean(), 2)
        })

    summary_df = pd.DataFrame(summary).sort_values(by="Mean Macro F1 (%)", ascending=False)

    print("\n" + "=" * 80)
    print("🏆 FINAL GPU BENCHMARK SUMMARY (Fold-Pure Repeated Stratified CV)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # 7. Save Artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "gpu_benchmark_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"💾 Results saved successfully to: {out_csv}")

    # LaTeX Table
    latex_path = os.path.join(args.out_dir, "table_results.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(summary_df.to_latex(index=False, caption="GPU Benchmark Performance of DNA-Inspired Requirements Classification", label="tab:gpu_results"))
    print(f"📄 LaTeX Table saved to: {latex_path}")

if __name__ == "__main__":
    main()
