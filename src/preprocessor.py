import re

class TextPreprocessor:
    """
    Standard preprocessor for requirement texts.
    Provides dual-mode cleaning:
    - normalize_for_transformer(): preserves casing and syntactic punctuation for SBERT.
    - clean_text(): normalizes and cleans for statistical lexical analysis (TF-IDF).
    - tokenize(): produces alphanumeric tokens for lexical/k-mer analysis.
    """
    @staticmethod
    def normalize_for_transformer(text: str) -> str:
        """
        Normalizes whitespace while preserving casing and sentence punctuation
        to retain maximum contextual information for Transformer encoders (SBERT).
        """
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans and normalizes input text for statistical TF-IDF vectorization.
        Converts to lowercase and removes non-alphanumeric noise.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters, keeping spaces, hyphens, and underscores
        text = re.sub(r'[^a-zA-Z0-9\s\-_]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def tokenize(text: str) -> list:
        """
        Cleans text and splits it into a list of alphanumeric tokens.
        """
        cleaned = TextPreprocessor.clean_text(text)
        tokens = [token for token in cleaned.split() if len(token) >= 2]
        return tokens
