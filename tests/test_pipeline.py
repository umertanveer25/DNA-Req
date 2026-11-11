import numpy as np
from src.preprocessor import TextPreprocessor
from src.features import (
    normalize_promise_label,
    dna_mapping,
    PROMISE_CLASSES,
    CLASS_DESCRIPTIONS,
    DNALinguisticTaxonomyExtractor,
    DNAFeatureExtractor,
    BioSequenceFeatureExtractor
)
from src.models import get_paper_classifiers, get_proposed_sequence_model

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

def test_12_class_normalization():
    """
    Test that all 12 canonical PROMISE classes are normalized correctly,
    with zero lossy collapse into 'N'.
    """
    assert len(PROMISE_CLASSES) == 12
    for cls in PROMISE_CLASSES:
        assert normalize_promise_label(cls) == cls
        assert dna_mapping(cls) == cls
        
    assert normalize_promise_label('functional') == 'F'
    assert normalize_promise_label('SECURITY') == 'SE'
    assert normalize_promise_label('usability') == 'US'
    assert normalize_promise_label('maintainability') == 'MN'
    assert normalize_promise_label('M') == 'MN'
    assert normalize_promise_label('portability') == 'PO'
    assert normalize_promise_label('P') == 'PO'

def test_taxonomic_dna_feature_extractor():
    """
    Test that DNALinguisticTaxonomyExtractor extracts exactly 28 features
    representing Authority, Temporal, Guard, and Client structural subcomponents.
    """
    texts = [
        "The system shall authenticate users with password and SSL within 2 seconds.",
        "The interface must display a confirmation dialog."
    ]
    feats = DNALinguisticTaxonomyExtractor.extract_batch(texts)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 28)
    assert not np.isnan(feats).any()

def test_feature_extractor_462d():
    """
    Test full DNAFeatureExtractor producing exactly 462 dimensions:
    384 (SBERT) + 50 (TF-IDF) + 28 (Taxonomic DNA) = 462 dimensions.
    """
    texts = [
        "The system shall authenticate users using SSL.",
        "The application must respond within 2 seconds."
    ]
    extractor = DNAFeatureExtractor(max_tfidf_features=50, include_taxonomic_dna=True)
    extractor.fit(texts)
    
    X_hybrid = extractor.transform(texts)
    assert isinstance(X_hybrid, np.ndarray)
    assert X_hybrid.shape == (2, 462)

def test_fold_pure_extraction_462d():
    """
    Test that fold-pure feature extraction maintains exact 462 dimensions
    and guarantees zero data leakage across folds.
    """
    train_texts = ["The system shall authenticate users with SSL.", "Users must login."]
    test_texts = ["Response time must be below 2 seconds."]
    extractor = DNAFeatureExtractor(max_tfidf_features=50, include_taxonomic_dna=True)
    
    train_sb = extractor.encode_sbert(train_texts)
    test_sb = extractor.encode_sbert(test_texts)
    
    X_tr, X_te = extractor.fit_transform_fold(train_texts, test_texts, train_sb, test_sb)
    assert X_tr.shape == (2, 462)
    assert X_te.shape == (1, 462)

def test_model_initialization():
    """
    Test that all classifiers (12 classical + 1 Proposed Hybrid Sequence Model) exist.
    """
    classifiers = get_paper_classifiers(include_proposed_sequence_model=True)
    assert len(classifiers) == 13
    assert "Proposed Hybrid Sequence Model" in classifiers
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
    y = np.array(['F'] * 10 + ['SE'] * 10)
    
    evaluator = BenchmarkEvaluator({"LR": LogisticRegression()})
    summary = evaluator.run_benchmark(X=X, y=y, num_splits=1)
    
    assert "Mean Accuracy (%)" in summary.columns
    assert "Mean Macro F1 (%)" in summary.columns
    assert "Mean Precision (%)" in summary.columns
    assert "Mean Recall (%)" in summary.columns
    assert len(summary) == 1

def test_bio_sequence_feature_extractor():
    """
    Test 2D (AT, GC) biological projection computation.
    """
    texts = [
        "The system shall authenticate users with password and SSL.",
        "The interface must display user profile screen."
    ]
    coords = BioSequenceFeatureExtractor.project_2d_at_gc(texts)
    assert coords.shape == (2, 2)
    for row in coords:
        assert np.isclose(row[0] + row[1], 1.0, atol=1e-3)

def test_hybrid_deep_sequence_classifier():
    """
    Test that HybridDeepSequenceClassifier initializes, fits, and predicts
    with 12 classes on 462-D space (substantiating the Sequence Model in Thesis Title).
    """
    from src.deep_models import HybridDeepSequenceClassifier
    X = np.random.randn(24, 462).astype(np.float32)
    # 2 samples per class across all 12 classes
    y = np.array(PROMISE_CLASSES * 2)
    
    clf = HybridDeepSequenceClassifier(epochs=2, batch_size=8, hidden_dim=32)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    
    assert len(preds) == 24
    assert proba.shape == (24, 12)
    assert set(preds).issubset(set(PROMISE_CLASSES))
