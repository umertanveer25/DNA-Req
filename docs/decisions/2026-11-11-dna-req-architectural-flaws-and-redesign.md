# Architecture Decision Record (ADR): Flaws and Strategic Redesign of DNA-Req

- **Status**: Proposed / Under Ruling
- **Date**: 2026-11-11
- **Context**: Comprehensive Architectural Flaw Analysis of `https://github.com/umertanveer25/DNA-Req`
- **Scope**: Feature Engineering, Bio-Inspired Modeling, Cross-Validation Purity, and Pipeline Reproducibility

---

## 1. Context & The "Don't Use Me" Gate (Step 0)

### Decision Classification
- **Class**: **Hard to reverse** (Methodological Foundation & Research Validity).
- **Justification**: The flaws identified in this architecture directly affect empirical validity, data integrity, mathematical correctness, and research credibility. Proceeding with the current architecture will guarantee rejection in peer review (IEEE TSE / ACM TOSEM) due to fatal cross-validation data leakage, cosmetic metaphorical mapping, mathematical misuse of probabilistic classifiers, and missing metric calculations.

---

## 2. Exhaustive Architectural Flaw Audit (Ground-Truth Code Inspection)

A forensic examination of the repository (`DNA-Req`) revealed 10 major architectural flaws:

### 🚨 Critical Flaws (Showstoppers for Research & Production)

1. **Fatal Data Leakage in Cross-Validation (`run_pipeline.py#L40-L42`, `src/evaluation.py#L30-L56`)**
   - **Flaw**: `extractor.fit_transform(X_text)` is executed globally across the entire dataset before any train/test splitting occurs.
   - **Impact**: `TfidfVectorizer(max_features=50)` fits vocabulary frequencies and IDF scores across all 969 samples simultaneously. Test set fold distributions leak directly into the feature space of the training folds. In academic peer review, this invalidates reported performance numbers (e.g., 83.38% SVM RBF).

2. **The "Bio-Inspired DNA" Metaphor Disconnect (`src/features.py#L6-L22`, `src/features.py#L62`)**
   - **Flaw**: The framework claims to be a "Bio-Inspired Machine Learning Framework" featuring "DNA-like sequences" and "dual-strand vectorization". In reality:
     - The feature vector is standard `[TF-IDF(50) || 1.5 * SBERT(384)]`.
     - Target classes are simply renamed via a Python dictionary: `F` $\to$ `A`, `US` $\to$ `T`, `SE` $\to$ `C`, `PE` $\to$ `G`, and others $\to$ `N`.
     - No biological computation, nucleotide sequence representation, k-mer tokenization, codon mapping, DNA language model (e.g., DNABERT), or evolutionary algorithm exists in the codebase.
   - **Impact**: Reviewers will immediately dismiss this as cosmetic rebranding ("cargo cult bio-inspiration").

3. **Target Class Collision & Information Annihilation (`src/features.py#L16-L22`, `data/Promise_Dataset.csv`)**
   - **Flaw**: In the PROMISE dataset, there is an existing target class named **`A`** (**Availability**, 31 instances). The mapper assigns Functional (`F`) to `A`, while mapping the real Availability class `A` to `N` (Neutral).
   - **Impact**: 8 real software requirement classes (`LF`, `A`, `MN`, `SC`, `FT`, `L`, `PO`, `O` totaling 248 instances) are collapsed into an undifferentiated composite class `N`. Furthermore, `generate_plots.py#L24` lists 12 original classes, causing immediate label mismatch with the 5 mapped classes (`A, T, C, G, N`).

4. **Mathematical Violation in Multinomial Naive Bayes (`src/evaluation.py#L25-L28`)**
   - **Flaw**: Continuous dense SBERT embeddings (which naturally range across positive and negative reals) are arbitrarily shifted by `X -= X.min()` and fed into `MultinomialNB`.
   - **Impact**: `MultinomialNB` assumes multinomial count distributions (discrete word counts). Shifting dense continuous vectors destroys geometric validity, explaining why Multinomial NB collapses to 18.78% Macro F1.

5. **Missing Benchmark Metrics & Paper Discrepancy (`src/evaluation.py#L55-L71` vs `README.md#L68-L82`)**
   - **Flaw**: `BenchmarkEvaluator` in `src/evaluation.py` only calculates `accuracy_score(y_test, preds)`. It **never computes Macro F1, Precision, or Recall**.
   - **Impact**: Table 1 in the README claims to report "Baseline Macro F1" derived from this 30x10 CV, but the repository code cannot reproduce its own published table.

### ⚠️ Significant Operational & Design Flaws

6. **Repeated CV Misrepresented as "Nested CV" (`src/evaluation.py#L7-L12`)**
   - The code executes `StratifiedKFold(n_splits=10)` repeated 30 times (Repeated K-Fold). Nested CV strictly requires an inner loop for hyperparameter optimization and an outer loop for evaluation.

7. **State Leakage / Object Mutation Across Folds (`src/evaluation.py#L46-L53`)**
   - The same dictionary of classifier instances is fitted across 300 folds without calling `sklearn.base.clone(clf)`, violating Scikit-learn best practices.

8. **Hardcoded File System Paths & Missing Files (Reproducibility Crash)**
   - `evaluate_fnfc.py` and `evaluate_fnfc_all.py` hardcode `C:/Users/umert/Downloads/FNFC.csv`.
   - `generate_plots.py` and `run_gb_parallel.py` require `results/X_features_cache.npy`, which is not committed.
   - Any external user cloning the repository encounters immediate fatal `FileNotFoundError` exceptions.

9. **Preprocessing Pipeline Degradation (`src/preprocessor.py`, `src/features.py`)**
   - `TextPreprocessor.clean_text` strips all punctuation and casing before feeding text into SBERT (`all-MiniLM-L6-v2`), actively degrading transformer positional and grammatical embeddings.
   - `TextPreprocessor.tokenize` is dead code.

10. **Arbitrary Magic Scaling Factor (`src/features.py#L62`)**
    - The `1.5 * SBERT` multiplier is hardcoded without grid-search, sensitivity curves, or ablation experiments.

---

## 3. The Three Competing Architectural Paths

To remediate these flaws and establish a scientifically defensible foundation, three distinct architectural paradigms are formulated:

### Option A: The Empirical ML Rigor Path (Clean Feature-Leakage Refactor)
* **Philosophy**: Fix all methodological flaws while preserving the lightweight, CPU-trainable classical ML stack.
* **Architecture**:
  1. Build a strict `sklearn.pipeline.Pipeline` or `FeatureUnion` where `TfidfVectorizer` and feature scalers are fitted **strictly inside each training fold**.
  2. Implement true `RepeatedStratifiedKFold` computing Accuracy, Balanced Accuracy, Macro F1, Precision, and Recall per fold.
  3. Replace the arbitrary `1.5` multiplier with a tunable hyperparameter or StandardScaler/Normalizer fusion.
  4. Decouple target classes properly: keep all 12 PROMISE classes or explicitly define an ablation for binary/5-class targets without collapsing genuine classes into `N`.
  5. Replace hardcoded local download paths with command-line arguments and downloadable datasets.
* **What it bets on**: Reviewers care most about clean methodology, zero data leakage, and reproducible classical ML benchmarks.
* **Strongest Downside**: It drops or downplays the "DNA" hook, reducing the paper/project novelty to standard classical NLP with SBERT + TF-IDF on PROMISE.

### Option B: The Genuine Bio-Inspired Sequence Modeling Path (True Genomic NLP)
* **Philosophy**: Transform the superficial metaphor into an actual, scientifically rigorous biological sequence algorithm.
* **Architecture**:
  1. **DNA Sequence Synthesizer**: Convert natural language requirements into synthetic pseudo-genomic sequences using rule-based nucleotide encoding (e.g., codon-based semantic mapping where syntactic POS tags or dependency triplets map to codon triplets `{ATG, GCT, TAA...}`).
  2. **Genomic Feature Extraction**: Extract genuine bio-computational metrics:
     - K-mer frequency spectrums ($k \in \{3, 4, 5, 6\}$).
     - GC-content, CpG island density, and sequence entropy.
     - Pairwise alignment scores via Smith-Waterman or Needleman-Wunsch against canonical requirement consensus motifs.
  3. **Biological Model Fusion**: Train genomic sequence representations alongside SBERT embeddings, demonstrating that biological sequence distance preserves semantic functional boundaries.
* **What it bets on**: High-novelty bio-computing intersection that genuinely delivers on the repository's title and README claims.
* **Strongest Downside**: High conceptual complexity. Reviewers might question why natural language should be converted to synthetic DNA sequences when modern NLP transformer tokenizers already excel at linguistic representations.

### Option C: The Modern Foundation Model & Adapter Path (Domain-Adapted Bio-NLP)
* **Philosophy**: Abandon classical 1990s ML algorithms (Decision Trees, Naive Bayes, KNN) on static vectors; adopt modern parameter-efficient fine-tuning (PEFT / LoRA) on transformer encoders.
* **Architecture**:
  1. Fine-tune a specialized domain encoder (e.g., `BioBERT`, `SciBERT`, or `ModernBERT-Req`) using LoRA on requirements classification.
  2. Implement an Adapter architecture that encodes functional vs non-functional constraints as orthogonal projection heads.
  3. Strict zero-shot cross-dataset evaluation across multiple requirement benchmarks (PROMISE, FNFC, PURE).
* **What it bets on**: Maximum classification accuracy, state-of-the-art Macro F1, and alignment with modern 2026 NLP research standards.
* **Strongest Downside**: Requires GPU acceleration, eliminates the lightweight CPU-based 12-classifier benchmark suite, and diverges completely from the original project's architecture.

---

## 4. Points of Convergence & Unexamined Defaults

### Where All Perspectives Agree
1. **The current data leakage is fatal**: Fitting TF-IDF globally before cross-validation must be eradicated immediately.
2. **The current DNA metaphor is indefensible in peer review**: Merely renaming 4 labels to `{A, T, C, G}` and dumping 8 classes into `N` while concatenating TF-IDF and SBERT is unacceptable for a paper claiming bio-inspired feature engineering.
3. **Reproducibility is currently broken**: Hardcoded paths (`C:/Users/umert/Downloads/FNFC.csv`) and missing cache files (`results/X_features_cache.npy`) must be replaced with robust asset management.
4. **Metrics must be overhauled**: Multi-class Macro F1, Cohen's Kappa, and per-class metrics must be computed inside the cross-validation loop.

### What Everyone Might Be Missing (The Unexamined Default)
* *The PROMISE dataset itself is heavily dated and known to have labeling noise and duplicate requirements.* All three approaches assume PROMISE (969 requirements) is a reliable benchmark. Without deduplication and rigorous noise auditing, any high accuracy (>83%) may simply reflect lexical overlap between train and test splits rather than true semantic generalization.

---

## 5. Cruxes & Fact Resolution

| # | Crux Fact | Verified Evidence | Consequence |
|---|---|---|---|
| **C1** | Does the code leak test data into TF-IDF? | **YES.** `run_pipeline.py#L40-L42`: `X_hybrid = extractor.fit_transform(X_text)` occurs before `evaluator.run_benchmark(X_hybrid, y)`. | TF-IDF vocabulary and IDF weights are contaminated by test folds. |
| **C2** | Does the code compute Macro F1 in 30x10 CV? | **NO.** `src/evaluation.py#L55`: only `accuracy_score` is computed. | The reported Macro F1 in Table 1 is unaccounted for in `run_pipeline.py`. |
| **C3** | Does PROMISE have an existing `A` class? | **YES.** `data/Promise_Dataset.csv` has 31 rows of class `A` (Availability). | Renaming Functional to `A` creates a collision that dumps actual Availability requirements into `N`. |
| **C4** | Can `MultinomialNB` accept negative inputs? | **NO.** Sklearn requires non-negative counts; shifting by `-min(X)` distorts embedding geometry. | Multinomial NB baseline is statistically invalid. |

---

## 6. Synthesis & Adjudication

### Conflict of Interest Disclosure
This analysis was performed via comprehensive static code audit across all Python modules, datasets, tests, and configuration files in the `DNA-Req` repository.

### Definitive Recommendation
Adopt a **Two-Stage Hybrid Remediation (Option A + Option B Light)**:

1. **Immediate Phase (Option A Refactor - Priority 1)**:
   - Encapsulate the feature pipeline inside `sklearn.pipeline.Pipeline` to guarantee that TF-IDF and scalers are fitted **strictly on training folds**.
   - Rewrite `src/evaluation.py` using `RepeatedStratifiedKFold(n_splits=10, n_repeats=30)` with `sklearn.base.clone(clf)`, logging Accuracy, Macro F1, Macro Precision, and Macro Recall.
   - Fix file paths: parameterize dataset inputs with fallback defaults and remove all machine-specific absolute paths.
   - Resolve the target class collision: maintain original requirement classes or explicitly rename mapped bases to avoid collision with standard NFR codes (e.g., use `BASE_A`, `BASE_T`, or explicit semantic tokens).

2. **Bio-Inspired Elevation (Option B Light - Priority 2)**:
   - If the research paper's core identity is "Bio-Inspired", implement genuine biological features (e.g., k-mer spectrum features, sequence entropy, or alignment motifs) so the biological contribution is algorithmic and mathematical, rather than merely nominal.

---

## 7. Concrete Revisit Trigger

Reopen this architectural decision if:
- **Trigger 1**: Cross-validation is re-run with strict fold isolation, and SVM RBF accuracy drops below **75%** (indicating the previous 83.38% was primarily driven by data leakage). In that event, pivot immediately to **Option C** (Transformer Fine-Tuning / LoRA).
- **Trigger 2**: A target venue (e.g., IEEE TSE) explicitly rejects bio-metaphorical framing in NLP; immediately drop Option B and publish under Option A as an empirical requirements engineering study.
