import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.features import (
    normalize_promise_label,
    PROMISE_CLASSES,
    CLASS_DESCRIPTIONS,
    DNAFeatureExtractor,
    BioSequenceFeatureExtractor
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 12-Class Confusion Matrices, ROC Curves, and Separability Plots")
    parser.add_argument("--promise_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE dataset")
    parser.add_argument("--fnfc_path", type=str, default="data/FNFC.csv", help="Path to FNFC dataset")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Output directory for plots")
    return parser.parse_args()

def plot_cm(y_true, y_pred, title, filepath, classes, display_labels):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"📊 Saved Confusion Matrix: {filepath}")

def plot_roc(y_true, y_proba, title, filepath, classes, display_labels):
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    plt.figure(figsize=(11, 8))
    cmap = plt.cm.get_cmap('tab20', n_classes)
    for i in range(n_classes):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], color=cmap(i), lw=1.8,
                     label=f'{display_labels[i]} (AUC = {roc_auc[i]:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"📈 Saved ROC AUC Curve: {filepath}")

def plot_at_gc_scatter(texts, y, filepath, classes, display_labels):
    """
    Generates 2D (AT, GC) biological projection scatter plot.
    """
    coords = BioSequenceFeatureExtractor.project_2d_at_gc(texts)
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', len(classes))
    
    for idx, (cls, label) in enumerate(zip(classes, display_labels)):
        mask = (y == cls)
        if np.any(mask):
            plt.scatter(coords[mask, 0], coords[mask, 1], 
                        color=cmap(idx), label=label, alpha=0.65, edgecolors='none', s=45)
            
    plt.title("2D (AT, GC) Bio-Symbolic Requirement Separability", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("AT Directive Ratio (Authority & Temporal Constraints)", fontsize=11)
    plt.ylabel("GC Directive Ratio (Guard & Client Constraints)", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"🧬 Saved 2D (AT, GC) Bio-Separability Plot: {filepath}")

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # 12 Canonical PROMISE Classes (No 'N' class)
    classes = PROMISE_CLASSES
    display_labels = [f"{c} ({CLASS_DESCRIPTIONS.get(c, c)})" for c in classes]

    # 1. Load PROMISE Dataset
    if not os.path.exists(args.promise_path):
        raise FileNotFoundError(f"PROMISE dataset not found at {args.promise_path}")
        
    print(f"[+] Loading PROMISE dataset from {args.promise_path}...")
    df_promise = pd.read_csv(args.promise_path)
    df_promise['Type'] = df_promise['Type'].str.strip()
    df_promise['Canonical_Class'] = df_promise['Type'].apply(normalize_promise_label)
    
    texts_promise = df_promise['Requirement'].tolist()
    y_promise = df_promise['Canonical_Class'].values

    # Check for cache or extract features on the fly
    cache_X = "results/X_features_cache.npy"
    cache_y = "results/y_labels_cache.npy"
    ext = DNAFeatureExtractor(max_tfidf_features=50, include_taxonomic_dna=True)
    
    # Check if cache exists and has 12 classes
    use_cache = False
    if os.path.exists(cache_X) and os.path.exists(cache_y):
        try:
            cached_y = np.load(cache_y, allow_pickle=True)
            if len(np.unique(cached_y)) == len(PROMISE_CLASSES):
                print(f"[+] Loading cached 12-class features from {cache_X}...")
                X_promise = np.load(cache_X)
                use_cache = True
        except Exception:
            use_cache = False
            
    if not use_cache:
        print("[+] Extracting DNA 462-D Hybrid Features (384 SBERT + 50 TF-IDF + 28 DNA)...")
        X_promise = ext.fit_transform(texts_promise)
        os.makedirs("results", exist_ok=True)
        np.save(cache_X, X_promise)
        np.save(cache_y, y_promise)
        print(f"[+] Cached 462-D features to {cache_X}")

    # 2. Split PROMISE (80/20 Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_promise, y_promise, test_size=0.20, random_state=42, stratify=y_promise
    )

    # 3. Train Classifier (SVM Linear or RBF)
    print("[+] Training SVM Linear on PROMISE training split (462-D)...")
    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Generate PROMISE Plots
    print("[+] Generating PROMISE Plots...")
    preds_promise = clf.predict(X_test)
    proba_promise = clf.predict_proba(X_test)
    
    plot_cm(y_test, preds_promise, 
            "Confusion Matrix - PROMISE Dataset 12-Class (SVM Linear)", 
            os.path.join(args.plots_dir, "cm_promise.png"), 
            classes, display_labels)
            
    plot_roc(y_test, proba_promise, 
             "Multi-Class ROC AUC - PROMISE Dataset (SVM Linear)", 
             os.path.join(args.plots_dir, "roc_promise.png"), 
             classes, display_labels)

    # 4b. 2D (AT, GC) Bio-Symbolic Separability Plot
    plot_at_gc_scatter(texts_promise, y_promise, 
                       os.path.join(args.plots_dir, "at_gc_separability.png"),
                       classes, display_labels)

    # 5. Handle FNFC Plots if dataset exists
    fnfc_candidate_paths = [args.fnfc_path, "data/FNFC.csv"]
    actual_fnfc_path = next((p for p in fnfc_candidate_paths if os.path.exists(p)), None)
    
    if actual_fnfc_path:
        print(f"[+] Found FNFC Dataset at {actual_fnfc_path}. Generating Zero-Shot Plots...")
        df_fnfc = pd.read_csv(actual_fnfc_path, encoding='ISO-8859-1')
        text_col = 'Requirement' if 'Requirement' in df_fnfc.columns else df_fnfc.columns[0]
        label_col = 'Type' if 'Type' in df_fnfc.columns else ('Class' if 'Class' in df_fnfc.columns else df_fnfc.columns[1])
        df_fnfc[label_col] = df_fnfc[label_col].astype(str).str.strip()
        df_fnfc['Canonical_Class'] = df_fnfc[label_col].apply(normalize_promise_label)

        # Evaluate on the classes common to PROMISE
        clf_full = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
        clf_full.fit(X_promise, y_promise)
        
        X_fnfc = ext.transform(df_fnfc[text_col].tolist())
        y_fnfc = df_fnfc['Canonical_Class'].values

        preds_fnfc = clf_full.predict(X_fnfc)
        proba_fnfc = clf_full.predict_proba(X_fnfc)

        plot_cm(y_fnfc, preds_fnfc, 
                "Confusion Matrix - FNFC Zero-Shot Generalizability", 
                os.path.join(args.plots_dir, "cm_fnfc.png"), 
                classes, display_labels)
                
        plot_roc(y_fnfc, proba_fnfc, 
                 "Multi-Class ROC AUC - FNFC Zero-Shot", 
                 os.path.join(args.plots_dir, "roc_fnfc.png"), 
                 classes, display_labels)
    else:
        print(f"[-] FNFC dataset not found. Skipping zero-shot plots.")

    print(f"\n[+] All requested plots generated successfully in '{args.plots_dir}/'!")

if __name__ == "__main__":
    main()
