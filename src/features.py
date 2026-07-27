import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

def dna_mapping(nfr_type: str) -> str:
    """
    DNA base mapping for software requirement classes (A, T, C, G, N).
    A: Functional (F)
    T: Usability (US)
    C: Security (SE)
    G: Performance (PE)
    N: Others (Default)
    """
    nfr_type = str(nfr_type).strip().upper()
    mapping = {
        'F': 'A',
        'US': 'T',
        'SE': 'C',
        'PE': 'G'
    }
    return mapping.get(nfr_type, 'N')

class DNAFeatureExtractor:
    """
    Extracts hybrid features by fusing statistical (TF-IDF) and 
    deep semantic (SBERT) representations.
    """
    def __init__(self, max_tfidf_features=2000, sbert_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the TF-IDF vectorizer and SBERT model.
        """
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 3),
            sublinear_tf=True
        )

    def fit(self, texts: list):
        """
        Fits the TF-IDF vectorizer on preprocessed text documents.
        """
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        self.tfidf_vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts: list) -> np.ndarray:
        """
        Fuses TF-IDF and SBERT features into a single DNA Fusion representation.
        """
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        
        # 1. Extract TF-IDF Statistical features
        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts).toarray()
        
        # Phase 0: Pure TF-IDF (No SBERT)
        return tfidf_features

    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fits and transforms in a single call.
        """
        self.fit(texts)
        return self.transform(texts)
