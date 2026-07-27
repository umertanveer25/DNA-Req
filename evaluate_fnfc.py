import os, sys, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

print("[+] Training SVM RBF on PROMISE dataset (100% of data)...")
df_train = pd.read_csv('data/Promise_Dataset.csv')
df_train['Type'] = df_train['Type'].str.strip()
df_train['DNA_Target'] = df_train['Type'].apply(dna_mapping)

ext = DNAFeatureExtractor(max_tfidf_features=50)
X_train = ext.fit_transform(df_train['Requirement'].tolist())
y_train = df_train['DNA_Target'].values

clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
clf.fit(X_train, y_train)

print("[+] Loading FNFC Dataset for Zero-Shot Evaluation...")
df_test = pd.read_csv('C:/Users/umert/Downloads/FNFC.csv', encoding='ISO-8859-1')
# Let's print columns to understand structure
print(f"FNFC Columns: {df_test.columns.tolist()}")

# Assuming standard columns 'Requirement' and 'Type' or 'Class'
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

print("="*50)
print(f"FNFC Generalizability Accuracy: {acc * 100:.2f}%")
print(f"FNFC Generalizability Macro F1: {f1 * 100:.2f}%")
print("="*50)
