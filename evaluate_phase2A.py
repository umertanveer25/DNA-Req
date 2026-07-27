import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
import random
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from src.features import dna_mapping, DNAFeatureExtractor

print("[+] Loading PROMISE dataset...")
df = pd.read_csv('data/Promise_Dataset.csv')
df['Type'] = df['Type'].str.strip()
df['DNA_Target'] = df['Type'].apply(dna_mapping)

print("[+] Loading FNFC Dataset for Zero-Shot Evaluation...")
df_test = pd.read_csv(r'C:\Users\umert\Downloads\FNFC.csv', encoding='ISO-8859-1')
text_col = 'Requirement' if 'Requirement' in df_test.columns else df_test.columns[0]
label_col = 'Type' if 'Type' in df_test.columns else ('Class' if 'Class' in df_test.columns else df_test.columns[1])
df_test[label_col] = df_test[label_col].astype(str).str.strip()
df_test['DNA_Target'] = df_test[label_col].apply(dna_mapping)

print("[+] Extracting 434-Dimension DNA Features (Phase 2-A: max_features=50, ngram=(1,3))...")
ext = DNAFeatureExtractor(max_tfidf_features=50)
X_train_full = ext.fit_transform(df['Requirement'].tolist())
y_train_full = df['DNA_Target'].values
X_test_full = ext.transform(df_test[text_col].tolist())
y_test_full = df_test['DNA_Target'].values

num_features = X_train_full.shape[1]
print(f"[+] Total DNA Features: {num_features}")

# ---------------------------------------------------------
# STAGE 1: GENETIC ALGORITHM (CUSTOM BIO-OPTIMIZER)
# ---------------------------------------------------------
print("\n[+] STAGE 1: Initializing Genetic Algorithm Feature Evolution...")

POPULATION_SIZE = 30
GENERATIONS = 10
MUTATION_RATE = 0.05
FITNESS_MODEL = SVC(kernel='linear', C=1, random_state=42) # Fast fitness model

def fitness_function(chromosome):
    """Evaluates a feature subset using internal 3-Fold CV"""
    # Chromosome is a boolean array
    mask = chromosome == 1
    if not np.any(mask): return 0.0 # Extinction
    
    X_subset = X_train_full[:, mask]
    
    # Internal fast 3-fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(FITNESS_MODEL, X_subset, y_train_full, cv=skf, scoring='f1_macro')
    return np.mean(scores)

# Initialize random population (1 = keep feature, 0 = drop feature)
population = [np.random.randint(2, size=num_features) for _ in range(POPULATION_SIZE)]

best_chromosome = None
best_fitness = 0.0

for gen in range(GENERATIONS):
    print(f"   [+] Evolving Generation {gen+1}/{GENERATIONS}...")
    
    # Evaluate fitness
    fitness_scores = []
    for chrom in population:
        fitness_scores.append(fitness_function(chrom))
    
    # Track best
    max_idx = np.argmax(fitness_scores)
    if fitness_scores[max_idx] > best_fitness:
        best_fitness = fitness_scores[max_idx]
        best_chromosome = population[max_idx].copy()
        
    print(f"       -> Best Fitness (Macro F1) in Gen {gen+1}: {best_fitness*100:.2f}% | Active Features: {np.sum(best_chromosome)}")
    
    # Selection (Tournament Selection)
    new_population = []
    for _ in range(POPULATION_SIZE):
        i, j = random.sample(range(POPULATION_SIZE), 2)
        parent1 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
        
        i, j = random.sample(range(POPULATION_SIZE), 2)
        parent2 = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
        
        # Crossover (Uniform)
        child = np.where(np.random.rand(num_features) > 0.5, parent1, parent2)
        
        # Mutation
        mutations = np.random.rand(num_features) < MUTATION_RATE
        child[mutations] = 1 - child[mutations]
        
        new_population.append(child)
        
    # Elitism: Keep the absolute best
    new_population[0] = best_chromosome.copy()
    population = new_population

print("\n[+] EVOLUTION COMPLETE.")
print(f"   -> Original Features: {num_features}")
print(f"   -> Optimized Features: {np.sum(best_chromosome)}")

# ---------------------------------------------------------
# STAGE 2: RIGOROUS EVALUATION (10-Fold CV)
# ---------------------------------------------------------
print("\n[+] STAGE 2: 10-Fold Evaluation on Optimized Features (PROMISE only)...")
mask = best_chromosome == 1
X_train_opt = X_train_full[:, mask]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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

print("\n[+] Testing Phase 2-A on PROMISE (10-Fold CV)...")
print("="*60)
print(f"{'Algorithm':<25} | {'Accuracy':<12} | {'Macro F1':<12}")
print("="*60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in algorithms.items():
    accs, f1s = [], []
    try:
        X_train_cv = X_train_opt
        if name == "Multinomial NB":
            scaler = MinMaxScaler()
            X_train_cv = scaler.fit_transform(X_train_opt)
            
        for tr, val in skf.split(X_train_cv, y_train_full):
            clf.fit(X_train_cv[tr], y_train_full[tr])
            preds = clf.predict(X_train_cv[val])
            accs.append(accuracy_score(y_train_full[val], preds))
            f1s.append(f1_score(y_train_full[val], preds, average='macro', zero_division=0))
            
        print(f"{name:<25} | {np.mean(accs)*100:>10.2f}% | {np.mean(f1s)*100:>10.2f}%", flush=True)
    except Exception as e:
        print(f"{name:<25} | Error: {str(e)[:40]}")

print("="*60)
