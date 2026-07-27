import os, sys, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

print("[+] Loading PROMISE dataset...")
df_train = pd.read_csv('data/Promise_Dataset.csv')
df_train['Type'] = df_train['Type'].str.strip()
df_train['DNA_Target'] = df_train['Type'].apply(dna_mapping)
y_train = df_train['DNA_Target'].values

print("[+] Loading FNFC Dataset for Zero-Shot Evaluation...")
df_test = pd.read_csv('C:/Users/umert/Downloads/FNFC.csv', encoding='ISO-8859-1')
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
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

for name, clf in algorithms.items():
    # GaussianNB doesn't support sparse matrices if TF-IDF is used, but DNAFeatureExtractor returns dense arrays
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average='macro', zero_division=0) * 100
        print(f"{name:<25} | {acc:>10.2f}% | {f1:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)
