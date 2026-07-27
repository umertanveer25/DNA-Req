import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['DNA_Target'] = df['Type'].apply(dna_mapping)
df = df[df['DNA_Target'] != 'N'].copy()

print("[+] Loading FNFC Dataset for Zero-Shot Evaluation...")
df_test = pd.read_csv(r'C:\Users\umert\Downloads\FNFC.csv', encoding='ISO-8859-1')
text_col = 'Requirement' if 'Requirement' in df_test.columns else df_test.columns[0]
label_col = 'Type' if 'Type' in df_test.columns else ('Class' if 'Class' in df_test.columns else df_test.columns[1])
df_test[label_col] = df_test[label_col].astype(str).str.strip()
df_test['DNA_Target'] = df_test[label_col].apply(dna_mapping)
df_test = df_test[df_test['DNA_Target'] != 'N'].copy()

print("[+] Extracting TF-IDF + SBERT Features (Phase 1A: max_features=50, ngram_range=(1,3))...")
ext = DNAFeatureExtractor(max_tfidf_features=50)
X_train = ext.fit_transform(df['Requirement'].tolist())
y_train = df['DNA_Target'].values
X_test = ext.transform(df_test[text_col].tolist())
y_test = df_test['DNA_Target'].values

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
    "Multinomial NB": MultinomialNB(),
    "Naive Bayes": GaussianNB()
}

print("\n[+] 1. Testing Phase 1A on PROMISE (10-Fold CV)...")
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in algorithms.items():
    accs, f1s = [], []
    try:
        # Scale for Multinomial NB if needed
        X_train_cv = X_train
        if name == "Multinomial NB":
            scaler = MinMaxScaler()
            X_train_cv = scaler.fit_transform(X_train)
            
        for tr, val in skf.split(X_train_cv, y_train):
            clf.fit(X_train_cv[tr], y_train[tr])
            preds = clf.predict(X_train_cv[val])
            accs.append(accuracy_score(y_train[val], preds))
            f1s.append(f1_score(y_train[val], preds, average='macro', zero_division=0))
            
        print(f"{name:<25} | {np.mean(accs)*100:>10.2f}% | {np.mean(f1s)*100:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)

print("\n[+] 2. Testing Phase 1A on FNFC (Zero-Shot Generalizability)...")
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

for name, clf in algorithms.items():
    try:
        if name == "Multinomial NB":
            scaler = MinMaxScaler()
            X_tr_scale = scaler.fit_transform(X_train)
            X_te_scale = scaler.transform(X_test)
            clf.fit(X_tr_scale, y_train)
            preds = clf.predict(X_te_scale)
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        print(f"{name:<25} | {acc*100:>10.2f}% | {f1*100:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)
