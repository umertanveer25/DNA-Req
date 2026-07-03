import re

class TextPreprocessor:
    """
    Standard preprocessor for requirement texts.
    Cleans punctuation, normalizes spacing, and tokenizes text.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans and normalizes the input text.
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
