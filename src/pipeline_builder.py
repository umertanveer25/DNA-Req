import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from src.features import DNAEncoderTransformer, CanonicalDNAEncoder

class HybridDNAFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Fuses DNA Codon features (fitted on train fold) with frozen SBERT embeddings.
    """
    def __init__(self, encoder_type='tfidf', max_codon_features=98, sbert_model_name='all-MiniLM-L6-v2', sbert_scale=1.5):
        self.encoder_type = encoder_type
        self.max_codon_features = max_codon_features
        self.sbert_model_name = sbert_model_name
        self.sbert_scale = sbert_scale
        
        if self.encoder_type == 'canonical':
            self.dna_encoder = CanonicalDNAEncoder(step_size=3)
        else:
            self.dna_encoder = DNAEncoderTransformer(max_features=self.max_codon_features)
            
        self.sbert_model = None

    def fit(self, X, y=None):
        self.dna_encoder.fit(X, y)
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(self.sbert_model_name)
        return self

    def transform(self, X):
        X_codons = self.dna_encoder.transform(X)
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(self.sbert_model_name)
        X_sbert = self.sbert_model.encode(X, show_progress_bar=False)
        return np.hstack((X_codons, X_sbert * self.sbert_scale))

def build_leak_free_pipeline(classifier, selector=None, encoder_type='canonical'):
    """
    Constructs a complete Scikit-Learn pipeline preventing all data leakage.
    """
    steps = [
        ('feature_fusion', HybridDNAFeatureExtractor(encoder_type=encoder_type)),
        ('scaler', MinMaxScaler())
    ]
    if selector is not None:
        steps.append(('feature_selection', selector))
    steps.append(('classifier', classifier))
    return Pipeline(steps)
