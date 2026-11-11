from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.deep_models import HybridDeepSequenceClassifier

def get_proposed_sequence_model(random_state=42):
    """
    Returns the proposed Hybrid Deep Sequence Model with BiLSTM + Self-Attention.
    Directly substantiates the PhD Thesis Title:
    'An Effective Hybrid Sequence Model for Automated Software Requirement 
     Classification Through Deep Representation Learning' (Synopsis [UT1], [UT2]).
    """
    return HybridDeepSequenceClassifier(
        epochs=20,
        batch_size=32,
        hidden_dim=128,
        dropout=0.3,
        random_state=random_state
    )

def get_paper_classifiers(random_state=42, include_proposed_sequence_model=True):
    """
    Returns the classifiers specified in the paper's benchmark suite.
    Defaults to include_proposed_sequence_model=True so the flagship PyTorch 
    BiLSTM + Self-Attention Hybrid Sequence Model is always benchmarked alongside baselines.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        "SVM Linear": SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ),
        "SVM RBF": SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50,
            random_state=random_state
        ),
        "Naive Bayes": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(
            n_neighbors=7
        ),
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5
        ),
        "KNN (k=3)": KNeighborsClassifier(
            n_neighbors=3
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight='balanced',
            random_state=random_state
        ),
        "Multinomial NB": MultinomialNB()
    }
    
    if include_proposed_sequence_model:
        models["Proposed Hybrid Sequence Model"] = get_proposed_sequence_model(random_state=random_state)
        
    return models
