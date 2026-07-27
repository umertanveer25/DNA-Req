import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

def dna_mapping(nfr_type: str) -> str:
    """
    DNA base mapping for software requirement classes.
    A: Functional (F)
    T: Usability (US)
    C: Security (SE)
    G: Performance (PE)
    N: Ambiguous/Unmapped Sequence (Default for minority classes)
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
    def __init__(self, max_tfidf_features=50, sbert_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the TF-IDF vectorizer and SBERT model.
        """
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
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
        
        # 2. Extract SBERT Semantic features
        sbert_embeddings = self.sbert_model.encode(cleaned_texts, show_progress_bar=False)
        
        # 3. DNA Hybrid Fusion (Concatenate TF-IDF and SBERT)
        X_hybrid = np.hstack((tfidf_features, sbert_embeddings * 1.5))
        return X_hybrid

    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fits and transforms in a single call.
        """
        self.fit(texts)
        return self.transform(texts)

class TextToDNAEncoder:
    """
    Translates raw English requirement text into literal DNA sequences (ATCGN).
    Words are mapped to their most highly correlated DNA base (A, T, C, G).
    Ambiguous, rare, or non-discriminatory words are assigned 'N'.
    The resulting DNA string is then processed using 3-mer (Codon) extraction.
    """
    def __init__(self, n_gram=3):
        self.word_to_base = {}
        # Codon extractor (3-mers of DNA bases)
        self.codon_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(n_gram, n_gram),
            sublinear_tf=True
        )

    def fit(self, texts, labels):
        from collections import defaultdict
        
        # We only care about mapping words to the 4 primary bases.
        # If a label is 'N', we skip using it for the base mapping dictionary.
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        
        for text, label in zip(texts, labels):
            if label == 'N':
                continue # Do not train word mappings on ambiguous requirements
                
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words): # Use set to count document frequency
                class_word_counts[label][w] += 1
                global_counts[w] += 1
                
        # Determine the best base for each word
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base[w] = 'N' # Too rare, ambiguous sequence
                continue
                
            # Find the class that has the highest frequency of this word
            max_class = 'N'
            max_freq = 0
            for cls in ['A', 'T', 'C', 'G']:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            
            # If the highest frequency is less than 50% of the total occurrences, it's ambiguous
            if (max_freq / total) < 0.5:
                self.word_to_base[w] = 'N'
            else:
                self.word_to_base[w] = max_class
                
        # Now fit the codon vectorizer on the translated DNA sequences
        dna_sequences = self._translate(texts)
        self.codon_vectorizer.fit(dna_sequences)
        return self

    def _translate(self, texts):
        dna_sequences = []
        for text in texts:
            words = TextPreprocessor.clean_text(text).split()
            sequence = "".join([self.word_to_base.get(w, 'N') for w in words])
            if not sequence:
                sequence = "N"
            dna_sequences.append(sequence)
        return dna_sequences

    def transform(self, texts):
        dna_sequences = self._translate(texts)
        # Return numerical features based on Codon (e.g. ATN, CCN) frequencies
        return self.codon_vectorizer.transform(dna_sequences).toarray()

    def fit_transform(self, texts, labels):
        self.fit(texts, labels)
        return self.transform(texts)

