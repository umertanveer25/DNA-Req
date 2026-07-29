import random
import os
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Sets global seeds across Python random, NumPy, and PyTorch for 
    complete end-to-end reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_reproducible_classifiers(seed: int = 42):
    """
    Returns fresh classifier instances with explicit random seeds.
    """
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB, GaussianNB

    return {
        "SVM RBF": SVC(kernel='rbf', C=10, gamma='scale', random_state=seed),
        "SVM Linear": SVC(kernel='linear', C=10, random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Multinomial NB": MultinomialNB(),
        "Naive Bayes": GaussianNB()
    }
