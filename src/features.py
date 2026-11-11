import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

def dna_mapping(nfr_type: str) -> str:
    """
    Symbolic 4-Base Genetic Target Mapping (Section 6.9, Eq. 3-4, [UT17]):
    A: Adenine   -> Functional Requirements (F)
    T: Thymine   -> Usability Requirements (US)
    C: Cytosine  -> Security Requirements (SE)
    G: Guanine   -> Performance Requirements (PE)
    N: Neutral   -> Other Non-Functional Requirements (LF, MN, SC, FT, L, PO, O, and Availability 'A')
    """
    nfr_type = str(nfr_type).strip().upper()
    mapping = {
        'F': 'A',
        'US': 'T',
        'SE': 'C',
        'PE': 'G'
    }
    return mapping.get(nfr_type, 'N')

class BioSequenceFeatureExtractor:
    """
    Auxiliary Bio-Inspired Feature Extractor (Section 6 & [UT27]).
    Maps requirements to 2D (AT, GC) nucleotide projections and auxiliary counts
    for domain interpretability and visual separability.
    """
    LEXICAL_BASE_KEYWORDS = {
        'A': ['shall', 'must', 'process', 'allow', 'system', 'provide', 'calculate', 'store', 'generate', 'execute'],
        'T': ['user', 'interface', 'display', 'screen', 'button', 'view', 'navigate', 'click', 'page', 'dialog', 'feedback', 'accessible'],
        'C': ['security', 'auth', 'password', 'ssl', 'access', 'encrypt', 'credential', 'privilege', 'permission', 'protect', 'firewall'],
        'G': ['seconds', 'latency', 'throughput', 'response', 'time', 'performance', 'bandwidth', 'speed', 'concurrent', 'transaction', 'load']
    }

    @staticmethod
    def extract_nucleotide_counts(text: str) -> dict:
        """
        Calculates bio-symbolic keyword activations for bases {A, T, C, G}.
        """
        cleaned = TextPreprocessor.clean_text(text)
        tokens = cleaned.split()
        total_tokens = max(len(tokens), 1)
        
        counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
        for token in tokens:
            for base, keywords in BioSequenceFeatureExtractor.LEXICAL_BASE_KEYWORDS.items():
                if any(kw in token for kw in keywords):
                    counts[base] += 1
        return counts

    @staticmethod
    def project_2d_at_gc(texts: list) -> np.ndarray:
        """
        Projects texts into a 2D (AT, GC) biological plane for interpretative visualization
        as committed in Synopsis Section 6 and Annotation [UT27].
        Returns array of shape (N, 2) where col 0 is AT-ratio and col 1 is GC-ratio.
        """
        projections = []
        for text in texts:
            counts = BioSequenceFeatureExtractor.extract_nucleotide_counts(text)
            total = sum(counts.values()) + 1e-5
            at_ratio = (counts['A'] + counts['T']) / total
            gc_ratio = (counts['G'] + counts['C']) / total
            projections.append([at_ratio, gc_ratio])
        return np.array(projections)

class DNAFeatureExtractor:
    """
    Extracts hybrid features by fusing statistical (TF-IDF) and 
    deep semantic (SBERT) representations.
    """
    def __init__(self, max_tfidf_features=50, sbert_model_name='all-MiniLM-L6-v2', gamma=1.5):
        """
        Initializes the TF-IDF vectorizer and SBERT model.
        gamma: empirical scaling factor for semantic embeddings (gamma=1.5 per Eq. 9).
        """
        self.gamma = gamma
        self.max_tfidf_features = max_tfidf_features
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def encode_sbert(self, texts: list) -> np.ndarray:
        """
        Extracts raw dense sentence embeddings via SBERT (all-MiniLM-L6-v2).
        Uses normalize_for_transformer() to preserve syntax, casing, and punctuation.
        Fixed representations; inference-only with zero label dependency.
        """
        transformer_texts = [TextPreprocessor.normalize_for_transformer(t) for t in texts]
        return self.sbert_model.encode(transformer_texts, show_progress_bar=False)

    def fit(self, texts: list):
        """
        Fits the TF-IDF vectorizer on preprocessed text documents.
        """
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        self.tfidf_vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts: list, sbert_embeddings: np.ndarray = None) -> np.ndarray:
        """
        Fuses TF-IDF and SBERT features into a single DNA Fusion representation.
        If sbert_embeddings is provided, reuses them to avoid redundant forward passes.
        """
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts).toarray()
        
        if sbert_embeddings is None:
            sbert_embeddings = self.sbert_model.encode(cleaned_texts, show_progress_bar=False)
            
        X_hybrid = np.hstack((tfidf_features, sbert_embeddings * self.gamma))
        return X_hybrid

    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fits and transforms in a single call.
        """
        self.fit(texts)
        return self.transform(texts)

    def fit_transform_fold(self, train_texts: list, test_texts: list, 
                           train_sbert: np.ndarray, test_sbert: np.ndarray) -> tuple:
        """
        Strict Fold-Pure Extraction:
        Fits TF-IDF exclusively on train_texts.
        Transforms both train_texts and test_texts.
        Fuses with respective pre-computed SBERT embeddings scaled by gamma.
        Guarantees zero test-fold vocabulary/frequency leakage into train space.
        """
        fold_tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        cleaned_train = [TextPreprocessor.clean_text(t) for t in train_texts]
        cleaned_test = [TextPreprocessor.clean_text(t) for t in test_texts]
        
        tr_tfidf = fold_tfidf.fit_transform(cleaned_train).toarray()
        te_tfidf = fold_tfidf.transform(cleaned_test).toarray()
        
        # If vocabulary size < max_tfidf_features, pad columns to guarantee exact expected dimension
        if tr_tfidf.shape[1] < self.max_tfidf_features:
            pad_width = self.max_tfidf_features - tr_tfidf.shape[1]
            tr_tfidf = np.pad(tr_tfidf, ((0, 0), (0, pad_width)), mode='constant')
            te_tfidf = np.pad(te_tfidf, ((0, 0), (0, pad_width)), mode='constant')
        
        X_tr = np.hstack((tr_tfidf, train_sbert * self.gamma))
        X_te = np.hstack((te_tfidf, test_sbert * self.gamma))
        return X_tr, X_te
