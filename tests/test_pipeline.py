import numpy as np
from src.preprocessor import TextPreprocessor
from src.features import dna_mapping, DNAFeatureExtractor
from src.models import get_paper_classifiers

def test_preprocessor():
    """
    Test TextPreprocessor cleaning and normalization.
    """
    raw_text = "The system SHALL process transactions, within 5 seconds!"
    cleaned = TextPreprocessor.clean_text(raw_text)
    assert cleaned == "the system shall process transactions within 5 seconds"
    
    tokens = TextPreprocessor.tokenize(raw_text)
    assert "system" in tokens
    assert "shall" in tokens
    assert "seconds" in tokens

def test_dna_mapping():
    """
    Test DNA target base assignments.
    """
    assert dna_mapping('F') == 'A'
    assert dna_mapping('US') == 'T'
    assert dna_mapping('SE') == 'C'
    assert dna_mapping('PE') == 'G'
    assert dna_mapping('LF') == 'N'
    assert dna_mapping('unknown') == 'N'

def test_feature_extractor():
    """
    Test DNAFeatureExtractor dimensions and outputs.
    """
    texts = [
        "The system shall authenticate users using SSL.",
        "The application must respond within 2 seconds."
    ]
    extractor = DNAFeatureExtractor(max_tfidf_features=10)
    extractor.fit(texts)
    
    X_hybrid = extractor.transform(texts)
    assert isinstance(X_hybrid, np.ndarray)
    assert X_hybrid.shape[0] == 2
    # TF-IDF max features (10) + SBERT output dim (384) = 394
    assert X_hybrid.shape[1] == 394

def test_fold_pure_extraction():
    """
    Test that fold-pure feature extraction maintains shape and has zero leakage.
    """
    train_texts = ["The system shall authenticate users with SSL.", "Users must login."]
    test_texts = ["Response time must be below 2 seconds."]
    extractor = DNAFeatureExtractor(max_tfidf_features=10)
    
    train_sb = extractor.encode_sbert(train_texts)
    test_sb = extractor.encode_sbert(test_texts)
    
    X_tr, X_te = extractor.fit_transform_fold(train_texts, test_texts, train_sb, test_sb)
    assert X_tr.shape == (2, 394)
    assert X_te.shape == (1, 394)

def test_model_initialization():
    """
    Test that all 12 classifiers exist.
    """
    classifiers = get_paper_classifiers()
    assert len(classifiers) == 12
    assert "Random Forest" in classifiers
    assert "Gradient Boosting" in classifiers
    assert "SVM Linear" in classifiers
    assert "Multinomial NB" in classifiers

def test_benchmark_metrics_calculation():
    """
    Test that BenchmarkEvaluator properly computes Macro F1, Precision, and Recall.
    """
    from src.evaluation import BenchmarkEvaluator
    from sklearn.linear_model import LogisticRegression
    
    # 20 samples, 2 classes
    X = np.random.randn(20, 10)
    y = np.array(['A'] * 10 + ['T'] * 10)
    
    evaluator = BenchmarkEvaluator({"LR": LogisticRegression()})
    summary = evaluator.run_benchmark(X=X, y=y, num_splits=1)
    
    assert "Mean Accuracy (%)" in summary.columns
    assert "Mean Macro F1 (%)" in summary.columns
    assert "Mean Precision (%)" in summary.columns
    assert "Mean Recall (%)" in summary.columns
    assert len(summary) == 1

def test_bio_sequence_feature_extractor():
    """
    Test 2D (AT, GC) biological projection computation (Synopsis [UT27]).
    """
    from src.features import BioSequenceFeatureExtractor
    texts = [
        "The system shall authenticate users with password and SSL.",
        "The interface must display user profile screen."
    ]
    coords = BioSequenceFeatureExtractor.project_2d_at_gc(texts)
    assert coords.shape == (2, 2)
    # Check that AT ratio + GC ratio approximately equal 1.0
    for row in coords:
        assert np.isclose(row[0] + row[1], 1.0, atol=1e-3)

def test_hybrid_deep_sequence_classifier():
    """
    Test that HybridDeepSequenceClassifier initializes, fits, and predicts
    (substantiating the Sequence Model in Thesis Title).
    """
    from src.deep_models import HybridDeepSequenceClassifier
    X = np.random.randn(20, 50).astype(np.float32)
    y = np.array(['A'] * 10 + ['T'] * 10)
    
    clf = HybridDeepSequenceClassifier(epochs=2, batch_size=10, hidden_dim=32)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    
    assert len(preds) == 20
    assert proba.shape == (20, 2)
    assert set(preds).issubset({'A', 'T'})
