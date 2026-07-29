# DNA-Inspired Software Requirements Classifier

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, bio-inspired machine learning framework that implements the exact methodology described in the research paper **"Bio-Inspired Feature Engineering for Enhanced Software Requirements Classification"**. 

This repository leverages biological metaphors by mapping software requirement classes into genetic DNA bases (A, T, C, G, N) and constructing a hybrid feature space fusing statistical TF-IDF keyword metrics with deep contextual SBERT sentence embeddings, optimized by bio-inspired meta-heuristic algorithms.

---

## 🧬 Methodology & Metaphorical Mapping

The framework encodes linguistic requirement rules into DNA-like sequences to build robust feature maps:

### 1. Symbolic DNA Base Target Mapping
Classes from the **PROMISE NFR dataset** are mapped into genetic bases:
*   **Adenine (A)**: Functional Requirements (F)
*   **Thymine (T)**: Usability Requirements (US)
*   **Guanine (G)**: Performance Requirements (PE)
*   **Cytosine (C)**: Security Requirements (SE)
*   **Neutral (N)**: All other Non-Functional Requirements (NFRs)

### 2. Hybrid DNA Feature Fusion
The input text space is transformed using a dual-strand vectorization approach:

$$\mathbf{X}_{\text{hybrid}} = \left[ \text{TF-IDF}(\mathbf{S}) \;\parallel\; 1.5 \times \text{SBERT}(\mathbf{S}) \right]$$

*   **Statistical Strand (TF-IDF)**: Standard word-level TF-IDF (DNA codon mappings, restricted to 98 features).
*   **Semantic Strand (SBERT)**: Dense 384-dimensional sentence embeddings generated via the `all-MiniLM-L6-v2` transformer model (scaled by a golden factor of 1.5 to emphasize semantic context).

---

## 🗺️ Phase-wise System Architecture

### 📊 Phase 0 & 2: The "Gold Standard" 14-Algorithm Evaluation
This evaluates the pure unoptimized representation learning approach, contrasting the Phase 0 (TF-IDF Baseline) against the Phase 2 (DNA + SBERT Hybrid Fusion Matrix) across 14 distinct classifiers (including ensembles) using rigorous 10-fold cross-validation.

| Rank | Algorithm | Phase 0 (TF-IDF Baseline) | Phase 2 (DNA + SBERT) | 🚀 Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **SVM RBF** | 79.46% | **86.06%** | `+ 6.60%` |
| **2** | **Logistic Regression** | 74.30% | **85.24%** | `+10.94%` |
| **3** | **SVM Linear** | 76.37% | **83.17%** | `+ 6.80%` |
| **4** | **KNN (k=5)** | 75.54% | **81.73%** | `+ 6.19%` |
| **5** | **KNN (k=3)** | 75.02% | **81.21%** | `+ 6.19%` |
| **6** | **Extra Trees** *(Ensemble)* | 76.26% | **80.91%** | `+ 4.65%` |
| **7** | **KNN (k=7)** | 76.37% | **80.90%** | `+ 4.53%` |
| **8** | **Gradient Boosting** *(Ensemble)* | 74.09% | **80.49%** | `+ 6.40%` |
| **9** | **Bagging Classifier** *(Ensemble)* | 72.50% | **79.80%** | `+ 7.30%` |
| **10** | **Random Forest** *(Ensemble)* | 72.75% | **79.35%** | `+ 6.60%` |
| **11** | **Multinomial NB** | 77.60% | **78.84%** | `+ 1.24%` |
| **12** | **AdaBoost** *(Ensemble)* | 51.80% | **70.59%** | `+18.79%` |
| **13** | **Naive Bayes (Gaussian)** | 59.03% | **65.84%** | `+ 6.81%` |
| **14** | **Decision Tree** | 62.13% | **62.44%** | `+ 0.31%` |

> **Conclusion**: 100% of the tested algorithms saw performance gains when transitioning from the TF-IDF baseline to the DNA+SBERT hybrid representation, with **SVM RBF** acting as the anchor model at **86.06%**.

### 🧬 Phase 2A - 2F: Bio-Inspired Feature Optimization (The Champion Phase)
In the final phase, we mathematically optimize the 482-dimensional fusion space using 6 rigorous bio-inspired meta-heuristic algorithms across a massive **30-split x 10-fold Cross-Validation** (3,300 model runs per optimizer).

#### 🏆 Ultimate Optimization Scoreboard (SVM RBF)

| Phase | Optimizer Algorithm | Selected Features | 30-Split Avg Accuracy |
| :---: | :--- | :---: | :---: |
| **2A** | 🥇 **Genetic Algorithm (GA)** | **239 / 482** | **86.84%** |
| **2C** | 🥈 **Ant Colony Optimization (ACO)** | 479 / 482 | 86.54% |
| **2F** | 🥉 **Whale Optimization Algorithm (WOA)** | 244 / 482 | 85.50% |
| **2E** | **Grey Wolf Optimizer (GWO)** | 257 / 482 | 85.42% |
| **2D** | **Artificial Bee Colony (ABC)** | 240 / 482 | 85.31% |
| **2B** | **Particle Swarm Optimization (PSO)** | 249 / 482 | 84.11% |

> **Conclusion**: The **Genetic Algorithm (Phase 2A)** is the absolute undisputed champion of feature selection for this architecture. By discarding over 50% of the dead-weight features (dropping from 482 to 239 dims), it successfully pushes the SVM RBF accuracy to an all-time high of **86.84%** without requiring *any* synthetic data balancing (like SMOTE).

---

## 🌍 Zero-Shot Generalizability (FNFC Dataset)

To mathematically prove that the DNA-inspired extractor is capturing underlying semantic principles rather than just memorizing the PROMISE dataset, we performed a strict **zero-shot cross-dataset evaluation**. 

The models were trained purely on the 969 PROMISE requirements, and tested blind on **7,060 totally unseen FNFC requirements**.

| Rank | Algorithm | Accuracy | Macro F1 | Notes |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 1 | **SVM RBF** | **77.58%** | **52.89%** | Best overall balance. Retains strong semantic generalization. |
| 🥈 2 | **Logistic Regression** | **76.93%** | **52.45%** | Highly robust semantic generalization. |

> **Conclusion**: Maintaining nearly 78% accuracy on a completely foreign dataset measuring 7,060 requirements demonstrates that the DNA feature extractor is robust to extreme domain shift and varying vocabulary.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/talktoumer94/DNA-Inspired-NFR-Classifier.git
cd DNA-Inspired-NFR-Classifier
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Bio-Optimized Phases
To run the winning Genetic Algorithm (Phase 2A) on all 11 classifiers:
```bash
python run_30_splits_phase2A.py
```

To run the Whale Optimization Algorithm (Phase 2F):
```bash
python run_30_splits_phase2F.py
```

---

## 📄 Citation
If you use this framework or reference our findings in your research, please cite our paper:
```bibtex
@article{tanveer2026bio,
  title={Bio-Inspired Feature Engineering for Enhanced Software Requirements Classification},
  author={Tanveer, Umer and Ali, Hashim},
  journal={IEEE Transactions on Software Engineering},
  year={2026}
}
```
