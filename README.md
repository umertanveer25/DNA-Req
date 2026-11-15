# DNA-Req: Multimodal Feature Engineering & Deep Sequence Classification for Automated Software Requirements Engineering

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance machine learning framework for automated software requirements classification across functional (FR) and fine-grained non-functional quality categories (NFR).

---

## 📌 Framework Highlights

1. **Multimodal 462-Dimensional Representation:**
   - **Semantic Strand (384-D):** Dense contextual sentence embeddings from Sentence-BERT (`all-MiniLM-L6-v2`).
   - **Lexical Strand (50-D):** Fold-isolated statistical n-grams with sublinear term scaling, fitted strictly on training folds to eliminate data leakage.
   - **DNA-Linguistic Taxonomy (28-D):** Deterministic syntactic decomposition encoding four structural pillars:
     - **Authority ($A$, 8-D):** Modal auxiliary verbs (`shall`, `must`, `will`, `should`, etc.).
     - **Temporal ($T$, 6-D):** Timing and frequency specifications (`seconds`, `ms`, `always`, `never`, etc.).
     - **Guard ($G$, 7-D):** Conditional delimiters and safety guards (`if`, `unless`, `whenever`, etc.).
     - **Client ($C$, 7-D):** Actor and entity identifiers (`user`, `admin`, `system`, `operator`, etc.).

2. **Flagship Deep Sequence Model:**
   - Bidirectional Long Short-Term Memory (BiLSTM, hidden dimension $64 \times 2 = 128$).
   - Multi-Head Scaled Dot-Product Self-Attention pooling ($d_a = 64$).
   - Dropout regularization ($p = 0.3$) and dense classification head.

3. **Multi-Corpus Validation ($N = 19,469$ Statements):**
   - **PROMISE:** 969 requirements across 12 canonical classes.
   - **PURE:** 11,440 sentences evaluated zero-shot under binary collapse ($F \to 0$, NFR $\to 1$).
   - **FNFC:** 7,060 requirements evaluated zero-shot across 14 fine-grained classes.

4. **Zero-Leakage Statistical Rigor:**
   - 10-fold Stratified Cross-Validation repeated across 30 randomized trials (3,600 estimator executions).
   - Validated via a formal 9-test non-parametric statistical battery (Wilcoxon, Friedman, Shapiro-Wilk, Holm-Bonferroni, Hedges' $g$).

---

## 📊 Framework Visualizations & Figures

| Figure | Description | Focus |
|:---|:---|:---|
| [`class_distribution.png`](class_distribution.png) | Class distribution across 12 canonical categories | PROMISE ($N = 969$) |
| [`method_overview.jpg`](method_overview.jpg) | End-to-end 462-D architectural workflow | Ingestion $\to$ 462-D $\to$ BiLSTM $\to$ Softmax |
| [`performance_comparison.png`](performance_comparison.png) | Head-to-head gains: TF-IDF (50-D) vs. 462-D Hybrid | 12 Classifiers comparison |
| [`confusion_matrix.png`](confusion_matrix.png) | Normalized 12-class confusion matrix | 10-fold CV on PROMISE |
| [`roc_promise.png`](roc_promise.png) | Multi-class Receiver Operating Characteristic curves | Area under curve $\ge 0.88$ |
| [`algorithm_ranking.png`](algorithm_ranking.png) | Forest plot ranking 13 evaluated models with 95% CIs | Performance ceiling ($82.77\%$) |
| [`cm_fnfc.png`](cm_fnfc.png) | Zero-shot multi-class transfer confusion matrix | FNFC ($N = 7,060$, 14 classes) |
| [`roc_fnfc.png`](roc_fnfc.png) | Zero-shot multi-class ROC curves | Cross-domain evaluation |

---

## 🔬 Formal 9-Test Statistical Validation Suite

| # | Statistical Evaluation Test | Test Statistic | Observed $p$-value | Formal Inference |
|:---:|:---|:---:|:---:|:---|
| 1 | **Shapiro-Wilk Normality Test** | $W = 0.8220$ | $p = 0.0168$ | Reject $H_0$ (Non-Gaussian, non-parametric required) |
| 2 | **Wilcoxon Signed-Rank Test** | $W = 69.00$ | $p = 0.00806$ | Reject $H_0$ ($p < 0.01$, highly significant gain) |
| 3 | **Friedman 4-Way Omnibus Test** | $\chi^2_{\text{F}} = 10.60$ | $p = 0.0141$ | Reject $H_0$ ($p < 0.05$, global ranking divergence) |
| 4 | **Holm-Bonferroni Step-Down** | $p_{\text{adj}}$ step-down | All $p_{\text{adj}} \le 0.0282$ | Family-wise false discoveries controlled |
| 5 | **Hedges' $g$ Effect Size** | $g_z = +0.7864$ | --- | Moderate-to-Large practical effect size |
| 6 | **Spearman Rank Correlation** | $\rho = 0.7552$ | $p = 0.0045$ | Statistically preserved relative rank ordering |
| 7 | **Generalization Gap Metric** | $\Delta = +19.94\%$ | $p = 0.00806$ | Significant out-of-domain transfer advantage |
| 8 | **Empirical Bootstrap (10,000 iter)** | Mean: $67.52\%$ | $95\%$ CI: $[55.37, 76.88]$ | Narrow bounds, low parameter dispersion |
| 9 | **Descriptive Dispersion Profile** | $\text{IQR} = 6.68\%$ | $\text{Median} = 74.94\%$ | Highly stable across randomized fold splits |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation
```bash
git clone https://github.com/umertanveer25/DNA-Req.git
cd DNA-Req
pip install -r requirements.txt
```

### 2. Run Repeated 10-Fold CV Pipeline (Zero Data Leakage)
Run in fast single-split demo mode:
```bash
python run_pipeline.py --demo
```
Run the full 30-repeat benchmark suite (3,600 model evaluations):
```bash
python run_pipeline.py
```

### 3. Run Out-of-Domain Generalization Benchmarks
Run 14-class zero-shot evaluation on FNFC ($N = 7,060$):
```bash
python run_multiclass_generalization.py
```
Run binary generalization on industrial PURE ($N = 11,440$):
```bash
python evaluate_pure.py
```

### 4. Run Statistical Significance Suite
```bash
python run_statistical_validation.py
```

### 5. Generate Publication Figures
```bash
python generate_all_updated_figures.py
```

### 6. Run Automated Unit Tests
```bash
python -m pytest tests/
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
