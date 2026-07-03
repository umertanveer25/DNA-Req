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
