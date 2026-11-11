import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.preprocessor import TextPreprocessor

# The 12 Canonical PROMISE Requirement Classes (Zero 'N' class)
PROMISE_CLASSES = ['F', 'SE', 'US', 'O', 'PE', 'LF', 'A', 'MN', 'SC', 'FT', 'L', 'PO']

CLASS_DESCRIPTIONS = {
    'F': 'Functional',
    'SE': 'Security',
    'US': 'Usability',
    'O': 'Operational',
    'PE': 'Performance',
    'LF': 'Look and Feel',
    'A': 'Availability',
    'MN': 'Maintainability',
    'SC': 'Scalability',
    'FT': 'Fault Tolerance',
    'L': 'Legal',
    'PO': 'Portability'
}

def normalize_promise_label(label: str) -> str:
    """
    Normalizes requirements classes across the 12 PROMISE taxonomies.
    Strictly eliminates artificial 'N' base; retains genuine 12 software requirement classes.
    """
    clean_label = str(label).strip().upper()
    aliases = {
        'FUNCTIONAL': 'F',
        'FR': 'F',
        'SECURITY': 'SE',
        'USABILITY': 'US',
        'OPERATIONAL': 'O',
        'PERFORMANCE': 'PE',
        'LOOK AND FEEL': 'LF',
        'AVAILABILITY': 'A',
        'MAINTAINABILITY': 'MN',
        'M': 'MN',
        'SCALABILITY': 'SC',
        'FAULT TOLERANCE': 'FT',
        'LEGAL': 'L',
        'PORTABILITY': 'PO',
        'P': 'PO'
    }
    canonical = aliases.get(clean_label, clean_label)
    if canonical in PROMISE_CLASSES:
        return canonical
    return clean_label

def dna_mapping(nfr_type: str) -> str:
    """
    Returns the normalized 12-class canonical PROMISE label.
    Preserves exact multi-class taxonomy without lossy 'N' collapse.
    """
    return normalize_promise_label(nfr_type)

class DNALinguisticTaxonomyExtractor:
    """
    Deterministic 28-Dimensional DNA-Linguistic Taxonomic Feature Extractor (Section 3.4).
    Reframes DNA strictly as a deterministic taxonomic mnemonic:
    - Authority (A): Mandatory legal/contractual directives (shall, must, will)
    - Temporal (T): Conditional triggers and timing constraints (when, after, seconds)
    - Guard (G): Security guards, error handling, access controls (encrypt, auth, exception)
    - Client (C): User-role interactions and interface directives (user, display, screen)
    """
    LEXICAL_BASE_KEYWORDS = {
        'A': ['shall', 'must', 'will', 'should', 'require', 'obligat', 'mandat', 'provid', 'allow', 'system'],
        'T': ['when', 'after', 'before', 'until', 'while', 'second', 'latency', 'timing', 'response', 'throughput', 'ms'],
        'G': ['security', 'auth', 'password', 'ssl', 'access', 'encrypt', 'credential', 'privilege', 'permission', 'protect', 'firewall', 'error', 'fail', 'backup'],
        'C': ['user', 'interface', 'display', 'screen', 'button', 'view', 'navigate', 'click', 'page', 'dialog', 'feedback', 'accessible', 'admin']
    }

    MODAL_WEIGHTS = {
        'shall': 1.0,
        'must': 0.9,
        'should': 0.6,
        'could': 0.3,
        'may': 0.2
    }

    DOMAIN_LEXICONS = {
        'SE': ['security', 'encrypt', 'authenticat', 'authoriz', 'privacy', 'confidential', 'integrity'],
        'RE': ['reliab', 'fault', 'toleran', 'failover', 'recovery', 'backup', 'redundant'],
        'PE': ['performance', 'latency', 'throughput', 'speed', 'bandwidth', 'load', 'capacity', 'concurrent'],
        'US': ['usability', 'accessible', 'interface', 'intuitive', 'workflow', 'ergonomic', 'visual'],
        'OP': ['operat', 'system', 'deploy', 'platform', 'config', 'environ', 'maintain', 'admin', 'schedul']
    }

    @staticmethod
    def extract_features(text: str) -> np.ndarray:
        """
        Extracts exactly 28 deterministic structural, modal, and taxonomic features:
        - 4 Base Counts (A, T, G, C)
        - 4 Base Ratios (counts / total tokens)
        - 5 Modal Directive Weights
        - 5 Syntactic Indicators
        - 5 Structural Complexity Metrics
        - 4 Domain Keyword Scores
        """
        text_lower = TextPreprocessor.clean_text(text)
        tokens = text_lower.split()
        num_tokens = max(len(tokens), 1)
        char_len = max(len(text), 1)

        # 1. Base Counts (4 dims: A, T, G, C)
        counts = {b: 0 for b in ['A', 'T', 'G', 'C']}
        for token in tokens:
            for b, kws in DNALinguisticTaxonomyExtractor.LEXICAL_BASE_KEYWORDS.items():
                if any(kw in token for kw in kws):
                    counts[b] += 1
        base_counts = [float(counts['A']), float(counts['T']), float(counts['G']), float(counts['C'])]

        # 2. Base Ratios (4 dims)
        base_ratios = [c / num_tokens for c in base_counts]

        # 3. Modal Directive Weights (5 dims)
        modal_scores = []
        for modal, w in DNALinguisticTaxonomyExtractor.MODAL_WEIGHTS.items():
            freq = tokens.count(modal)
            modal_scores.append(float(freq * w))

        # 4. Syntactic Indicators (5 dims)
        # Passive voice indicator proxy ('is/was/be' + past participle ending 'ed')
        passive = 1.0 if re.search(r'\b(is|was|are|were|be|been|being)\s+\w+ed\b', text_lower) else 0.0
        # Conditional clause ('if', 'when', 'unless', 'in case')
        conditional = 1.0 if re.search(r'\b(if|when|unless|provided that)\b', text_lower) else 0.0
        # Numeric quantitative specs (e.g., '99.9%', '2 seconds', '500 users')
        numeric = 1.0 if re.search(r'\b\d+(\.\d+)?%?\b', text) else 0.0
        # Technical acronyms (UPPERCASE tokens of length 2-5)
        acronyms = float(len(re.findall(r'\b[A-Z]{2,5}\b', text)))
        # Punctuation density (commas, semicolons, hyphens)
        punct_density = float(len(re.findall(r'[,;:\-]', text))) / num_tokens
        syntactic_indicators = [passive, conditional, numeric, acronyms, punct_density]

        # 5. Structural Complexity Metrics (5 dims)
        total_chars = float(char_len)
        total_tokens = float(num_tokens)
        avg_word_len = total_chars / total_tokens
        parse_depth_proxy = float(text.count(',') + text.count(';') + text.count('(') + 1)
        # Automated readability index proxy: 4.71*(chars/words) + 0.5*(words/sentences) - 21.43
        sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
        ari_proxy = 4.71 * (total_chars / total_tokens) + 0.5 * (total_tokens / sentences) - 21.43
        structural_metrics = [total_chars, total_tokens, avg_word_len, parse_depth_proxy, ari_proxy]

        # 6. Domain Keyword Scores (4 dims: Security, Reliability, Performance, Usability)
        domain_scores = []
        for dom, kws in DNALinguisticTaxonomyExtractor.DOMAIN_LEXICONS.items():
            score = sum(1.0 for token in tokens if any(kw in token for kw in kws))
            domain_scores.append(score)

        # Concatenate all 28 dimensions
        feature_vector = np.array(base_counts + base_ratios + modal_scores + 
                                  syntactic_indicators + structural_metrics + domain_scores, dtype=np.float32)
        return feature_vector

    @staticmethod
    def extract_batch(texts: list) -> np.ndarray:
        return np.array([DNALinguisticTaxonomyExtractor.extract_features(t) for t in texts], dtype=np.float32)

class BioSequenceFeatureExtractor:
    """
    2D (AT, GC) Bio-Symbolic Projection (Section 6 & [UT27]).
    """
    @staticmethod
    def project_2d_at_gc(texts: list) -> np.ndarray:
        dna_feats = DNALinguisticTaxonomyExtractor.extract_batch(texts)
        # Columns 0, 1, 2, 3 correspond to counts [A, T, G, C]
        counts_a = dna_feats[:, 0]
        counts_t = dna_feats[:, 1]
        counts_g = dna_feats[:, 2]
        counts_c = dna_feats[:, 3]
        total = counts_a + counts_t + counts_g + counts_c + 1e-5
        
        at_ratio = (counts_a + counts_t) / total
        gc_ratio = (counts_g + counts_c) / total
        return np.column_stack((at_ratio, gc_ratio))

class DNAFeatureExtractor:
    """
    Extracts high-dimensional hybrid features fusing:
    - Tier 1: Dense contextual SBERT representations (384 dimensions)
    - Tier 2: Lexical syntactic TF-IDF n-grams (50 dimensions)
    - Tier 3: DNA-Linguistic Taxonomic Rules (28 dimensions)
    Total combined dimension: 384 + 50 + 28 = 462 dimensions (matching Table 2 of paper).
    """
    def __init__(self, max_tfidf_features=50, sbert_model_name='all-MiniLM-L6-v2', 
                 gamma=1.5, include_taxonomic_dna=True):
        self.gamma = gamma
        self.include_taxonomic_dna = include_taxonomic_dna
        self.max_tfidf_features = max_tfidf_features
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )

    def encode_sbert(self, texts: list) -> np.ndarray:
        transformer_texts = [TextPreprocessor.normalize_for_transformer(t) for t in texts]
        return self.sbert_model.encode(transformer_texts, show_progress_bar=False)

    def fit(self, texts: list):
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        self.tfidf_vectorizer.fit(cleaned_texts)
        return self

    def transform(self, texts: list, sbert_embeddings: np.ndarray = None) -> np.ndarray:
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]
        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts).toarray()
        
        # Pad to max_tfidf_features if vocabulary size is smaller
        if tfidf_features.shape[1] < self.max_tfidf_features:
            pad_width = self.max_tfidf_features - tfidf_features.shape[1]
            tfidf_features = np.pad(tfidf_features, ((0, 0), (0, pad_width)), mode='constant')
            
        if sbert_embeddings is None:
            sbert_embeddings = self.encode_sbert(texts)
            
        parts = [tfidf_features, sbert_embeddings * self.gamma]
        if self.include_taxonomic_dna:
            taxonomic_feats = DNALinguisticTaxonomyExtractor.extract_batch(texts)
            parts.append(taxonomic_feats)
            
        X_hybrid = np.hstack(parts)
        return X_hybrid

    def fit_transform(self, texts: list) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def fit_transform_fold(self, train_texts: list, test_texts: list, 
                           train_sbert: np.ndarray, test_sbert: np.ndarray) -> tuple:
        """
        Strict Fold-Pure Feature Extraction:
        Fits TF-IDF exclusively on train_texts.
        Transforms both train and test texts.
        Fuses with respective SBERT and deterministic 28-d taxonomic DNA features.
        Zero data leakage across folds.
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
        
        if tr_tfidf.shape[1] < self.max_tfidf_features:
            pad_width = self.max_tfidf_features - tr_tfidf.shape[1]
            tr_tfidf = np.pad(tr_tfidf, ((0, 0), (0, pad_width)), mode='constant')
            te_tfidf = np.pad(te_tfidf, ((0, 0), (0, pad_width)), mode='constant')
        
        tr_parts = [tr_tfidf, train_sbert * self.gamma]
        te_parts = [te_tfidf, test_sbert * self.gamma]
        
        if self.include_taxonomic_dna:
            tr_dna = DNALinguisticTaxonomyExtractor.extract_batch(train_texts)
            te_dna = DNALinguisticTaxonomyExtractor.extract_batch(test_texts)
            tr_parts.append(tr_dna)
            te_parts.append(te_dna)
            
        X_tr = np.hstack(tr_parts)
        X_te = np.hstack(te_parts)
        return X_tr, X_te
