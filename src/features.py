import numpy as np
import itertools
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessor import TextPreprocessor

class CanonicalDNAEncoder(BaseEstimator, TransformerMixin):
    """
    Strict 4-Base Canonical DNA Encoder.
    Restricts codon feature space strictly to the 64 canonical DNA codons (A, T, C, G).
    Ignores any n-gram windows containing 'N' or non-canonical characters.
    Enforces non-overlapping Open Reading Frame (ORF) translation (step_size=3).
    """
    CANONICAL_BASES = ['A', 'T', 'C', 'G']
    CANONICAL_CODONS = ["".join(p) for p in itertools.product(CANONICAL_BASES, repeat=3)]
    CODON_TO_IDX = {codon: i for i, codon in enumerate(CANONICAL_CODONS)}

    def __init__(self, step_size=3, sublinear_tf=True):
        self.step_size = step_size
        self.sublinear_tf = sublinear_tf
        self.word_to_base_ = {}

    def fit(self, texts, labels):
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        
        for text, label in zip(texts, labels):
            if label == 'N':
                continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1
                
        self.word_to_base_ = {}
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base_[w] = 'N'
                continue
            max_class, max_freq = 'N', 0
            for cls in self.CANONICAL_BASES:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            
            if (max_freq / total) >= 0.5:
                self.word_to_base_[w] = max_class
            else:
                self.word_to_base_[w] = 'N'
        return self

    def _extract_canonical_codons(self, text):
        words = TextPreprocessor.clean_text(text).split()
        bases = [self.word_to_base_.get(w, 'N') for w in words]
        
        codons = []
        # Enforce non-overlapping Reading Frame (step_size=3)
        for i in range(0, len(bases) - 2, self.step_size):
            triplet = "".join(bases[i:i+3])
            # Strictly validate that all 3 bases belong to {A, T, C, G}
            if all(b in self.CANONICAL_BASES for b in triplet):
                codons.append(triplet)
        return codons

    def transform(self, texts):
        X = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            codons = self._extract_canonical_codons(text)
            for codon in codons:
                idx = self.CODON_TO_IDX[codon]
                X[i, idx] += 1.0
                
        if self.sublinear_tf:
            X = np.where(X > 0, 1.0 + np.log(X), 0.0)
            
        # L2 normalize feature vectors
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, texts, labels):
        return self.fit(texts, labels).transform(texts)


class IUPACDegenerateDNAEncoder(BaseEstimator, TransformerMixin):
    """
    IUPAC-Compliant Degenerate DNA Codon Encoder.
    Expands ambiguous 'N' nucleotides into uniform probability distributions over canonical bases.
    Maps sequences onto the 64 canonical codon probability simplex.
    """
    CANONICAL_BASES = ['A', 'T', 'C', 'G']
    CANONICAL_CODONS = ["".join(p) for p in itertools.product(CANONICAL_BASES, repeat=3)]
    CODON_TO_IDX = {codon: i for i, codon in enumerate(CANONICAL_CODONS)}
    
    NUC_PROB = {
        'A': {'A': 1.0, 'T': 0.0, 'C': 0.0, 'G': 0.0},
        'T': {'A': 0.0, 'T': 1.0, 'C': 0.0, 'G': 0.0},
        'C': {'A': 0.0, 'T': 0.0, 'C': 1.0, 'G': 0.0},
        'G': {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 1.0},
        'N': {'A': 0.25, 'T': 0.25, 'C': 0.25, 'G': 0.25} # IUPAC uniform ambiguity distribution
    }

    def __init__(self, step_size=3):
        self.step_size = step_size
        self.word_to_base_ = {}

    def fit(self, texts, labels):
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        for text, label in zip(texts, labels):
            if label == 'N': continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1
        self.word_to_base_ = {}
        for w, total in global_counts.items():
            if total < 3:
                self.word_to_base_[w] = 'N'
                continue
            max_class, max_freq = 'N', 0
            for cls in self.CANONICAL_BASES:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
            self.word_to_base_[w] = max_class if (max_freq / total) >= 0.5 else 'N'
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), 64), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            words = TextPreprocessor.clean_text(text).split()
            bases = [self.word_to_base_.get(w, 'N') for w in words]
            
            for i in range(0, len(bases) - 2, self.step_size):
                b1, b2, b3 = bases[i], bases[i+1], bases[i+2]
                # Expand triplet probabilistically across 64 canonical codons
                for c1, p1 in self.NUC_PROB[b1].items():
                    if p1 == 0: continue
                    for c2, p2 in self.NUC_PROB[b2].items():
                        if p2 == 0: continue
                        for c3, p3 in self.NUC_PROB[b3].items():
                            if p3 == 0: continue
                            codon = c1 + c2 + c3
                            X[row_idx, self.CODON_TO_IDX[codon]] += p1 * p2 * p3

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms


class BiologicalAminoAcidEncoder(BaseEstimator, TransformerMixin):
    """
    Biological Amino Acid Translator.
    Translates 64 canonical DNA codons into 20 standard amino acids + 1 Stop codon
    using NCBI Standard Genetic Code Table 1.
    """
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    AMINO_ACIDS = sorted(list(set(CODON_TABLE.values())))
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    def __init__(self, step_size=3):
        self.step_size = step_size
        self.canonical_encoder = CanonicalDNAEncoder(step_size=step_size, sublinear_tf=False)

    def fit(self, texts, labels):
        self.canonical_encoder.fit(texts, labels)
        return self

    def transform(self, texts):
        X_aa = np.zeros((len(texts), len(self.AMINO_ACIDS)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            codons = self.canonical_encoder._extract_canonical_codons(text)
            for codon in codons:
                aa = self.CODON_TABLE[codon]
                X_aa[row_idx, self.AA_TO_IDX[aa]] += 1.0
                
        norms = np.linalg.norm(X_aa, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X_aa / norms


class DNAEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compliant DNA Codon Feature Extractor.
    Learns word-to-base mappings and TF-IDF codon statistics strictly 
    from training data during fit(), preventing data leakage across CV folds.
    """
    def __init__(self, n_gram=3, max_features=98, min_word_count=3, base_ratio_thresh=0.5):
        self.n_gram = n_gram
        self.max_features = max_features
        self.min_word_count = min_word_count
        self.base_ratio_thresh = base_ratio_thresh
        
        self.word_to_base_ = {}
        self.codon_vectorizer_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("DNAEncoderTransformer requires target labels y for supervised base mapping.")
            
        class_word_counts = defaultdict(lambda: defaultdict(int))
        global_counts = defaultdict(int)
        
        # 1. Compute class-word frequencies strictly on training fold
        for text, label in zip(X, y):
            if label == 'N':
                continue
            words = TextPreprocessor.clean_text(text).split()
            for w in set(words):
                class_word_counts[label][w] += 1
                global_counts[w] += 1

        self.word_to_base_ = {}
        canonical_bases = ['A', 'T', 'C', 'G']
        
        for w, total in global_counts.items():
            if total < self.min_word_count:
                self.word_to_base_[w] = 'N'
                continue
            max_class, max_freq = 'N', 0
            for cls in canonical_bases:
                if class_word_counts[cls][w] > max_freq:
                    max_freq = class_word_counts[cls][w]
                    max_class = cls
                    
            if (max_freq / total) < self.base_ratio_thresh:
                self.word_to_base_[w] = 'N'
            else:
                self.word_to_base_[w] = max_class

        # 2. Translate training texts to DNA sequences and fit TF-IDF vectorizer
        dna_seqs = self._translate(X)
        self.codon_vectorizer_ = TfidfVectorizer(
            analyzer='char',
            ngram_range=(self.n_gram, self.n_gram),
            max_features=self.max_features,
            sublinear_tf=True
        )
        self.codon_vectorizer_.fit(dna_seqs)
        return self

    def _translate(self, X):
        dna_sequences = []
        for text in X:
            words = TextPreprocessor.clean_text(text).split()
            seq = "".join([self.word_to_base_.get(w, 'N') for w in words])
            dna_sequences.append(seq if seq else "N")
        return dna_sequences

    def transform(self, X):
        if self.codon_vectorizer_ is None:
            raise RuntimeError("DNAEncoderTransformer must be fitted before calling transform().")
        dna_seqs = self._translate(X)
        return self.codon_vectorizer_.transform(dna_seqs).toarray()
