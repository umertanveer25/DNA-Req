import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.features import dna_mapping, TextToDNAEncoder

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()

# Target Mapping (Keep N in the dataset as an unmapped sequence target)
df['DNA_Target'] = df['Type'].apply(dna_mapping)

print("[+] Extracting Amino Acid Sequences (ATN, CCN, NNN)...")
ext = TextToDNAEncoder(n_gram=3)

# We fit the encoder using the text and labels. 
# The encoder will assign 'N' to ambiguous words, and 'A','T','C','G' to strong indicators.
X_train = ext.fit_transform(df['Requirement'].tolist(), df['DNA_Target'].tolist())
y_train = df['DNA_Target'].values

print(f"\n[+] Translated English to DNA Sequences! Extracted {X_train.shape[1]} unique codons.")
# Show an example of the translation
sample_idx = 42
sample_text = df['Requirement'].iloc[sample_idx]
sample_dna = ext._translate([sample_text])[0]
print(f"Sample English : {sample_text}")
print(f"Sample DNA Seq : {sample_dna}")

print("\n[+] 1. Testing Phase 3 (Amino Acid Encoding) on PROMISE (10-Fold CV)...")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler

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

print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in algorithms.items():
    accs, f1s = [], []
    try:
        for tr, val in skf.split(X_train, y_train):
            clf.fit(X_train[tr], y_train[tr])
            preds = clf.predict(X_train[val])
            accs.append(accuracy_score(y_train[val], preds))
            f1s.append(f1_score(y_train[val], preds, average='macro', zero_division=0))
            
        print(f"{name:<25} | {np.mean(accs)*100:>10.2f}% | {np.mean(f1s)*100:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)
