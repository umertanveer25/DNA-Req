import os, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

# Create plots directory
os.makedirs('plots', exist_ok=True)

# 1. Load PROMISE
print("[+] Loading Cached PROMISE dataset features...")
X_promise = np.load('results/X_features_cache.npy')
y_promise = np.load('results/y_labels_cache.npy', allow_pickle=True)

classes = ['F', 'A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US']

# 2. Load FNFC
print("[+] Loading FNFC Dataset...")
df_fnfc = pd.read_csv('C:/Users/umert/Downloads/FNFC.csv', encoding='ISO-8859-1')
text_col = 'Requirement' if 'Requirement' in df_fnfc.columns else df_fnfc.columns[0]
label_col = 'Type' if 'Type' in df_fnfc.columns else ('Class' if 'Class' in df_fnfc.columns else df_fnfc.columns[1])
df_fnfc[label_col] = df_fnfc[label_col].astype(str).str.strip()
df_fnfc['DNA_Target'] = df_fnfc[label_col].apply(dna_mapping)

print("[+] Extracting DNA Features for FNFC...")
ext = DNAFeatureExtractor(max_tfidf_features=50)
df_promise = pd.read_csv('data/Promise_Dataset.csv')
ext.fit(df_promise['Requirement'].tolist()) # Fit tfidf
X_fnfc = ext.transform(df_fnfc[text_col].tolist())
y_fnfc = df_fnfc['DNA_Target'].values

# Split PROMISE for plotting ROC
X_train, X_test, y_train, y_test = train_test_split(X_promise, y_promise, test_size=0.2, random_state=42, stratify=y_promise)

# 4. Train SVM RBF (Probability=True for ROC)
print("[+] Training SVM RBF...")
clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
clf.fit(X_train, y_train)

# --- Plotting Functions ---
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300)
    plt.close()

def plot_roc(y_true, y_proba, title, filename):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        # Only plot if class exists in true labels
        if np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    for i, color in zip(range(n_classes), colors):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300)
    plt.close()

# 5. Generate PROMISE Plots
print("[+] Generating PROMISE Plots...")
preds_promise = clf.predict(X_test)
proba_promise = clf.predict_proba(X_test)
plot_cm(y_test, preds_promise, "Confusion Matrix - PROMISE Dataset (SVM RBF)", "cm_promise")
plot_roc(y_test, proba_promise, "ROC AUC Curves - PROMISE Dataset (SVM RBF)", "roc_promise")

# 6. Generate FNFC Plots
# Train on FULL promise for FNFC generalizability test
clf_full = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
clf_full.fit(X_promise, y_promise)

print("[+] Generating FNFC Generalizability Plots...")
preds_fnfc = clf_full.predict(X_fnfc)
proba_fnfc = clf_full.predict_proba(X_fnfc)
plot_cm(y_fnfc, preds_fnfc, "Confusion Matrix - FNFC Generalizability (SVM RBF)", "cm_fnfc")

# Handle FNFC ROC gracefully in case not all 12 classes are present
try:
    plot_roc(y_fnfc, proba_fnfc, "ROC AUC Curves - FNFC Generalizability (SVM RBF)", "roc_fnfc")
except Exception as e:
    print(f"[-] Could not plot FNFC ROC (likely missing classes): {str(e)}")

print("[+] All plots generated in the 'plots/' directory!")
