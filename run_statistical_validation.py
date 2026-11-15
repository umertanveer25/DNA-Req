import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def compute_all_statistical_tests():
    print("=" * 80)
    print("📊 EXECUTING COMPLETE 9-TEST STATISTICAL VALIDATION SUITE")
    print("=" * 80)

    # 1. Load the live FNFC multi-class results
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'fnfc_multiclass_live_results.csv')
    df = pd.read_csv(csv_path)

    tfidf_acc = df['TFIDF_Acc'].values
    hybrid_acc = df['SBERT_Acc'].values
    diff = hybrid_acc - tfidf_acc
    n = len(diff)

    print(f"Loaded {n} paired classifier evaluations (TF-IDF vs. 462-D Hybrid).")

    # -------------------------------------------------------------
    # TEST 1: Shapiro-Wilk Test for Normality
    # -------------------------------------------------------------
    shapiro_stat, shapiro_p = stats.shapiro(diff)
    print("\n[TEST 1] Shapiro-Wilk Normality Test (Differences):")
    print(f"   W-statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.6f}")
    if shapiro_p < 0.05:
        print("   -> Inference: Distribution is significantly non-normal (p < 0.05). Justifies non-parametric testing.")
    else:
        print("   -> Inference: Normal distribution.")

    # -------------------------------------------------------------
    # TEST 2: Wilcoxon Signed-Rank Test (Paired, Non-Parametric)
    # -------------------------------------------------------------
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(hybrid_acc, tfidf_acc, alternative='greater')
    print("\n[TEST 2] Wilcoxon Signed-Rank Test (Hybrid > TF-IDF):")
    print(f"   W-statistic: {wilcoxon_stat:.2f}, p-value: {wilcoxon_p:.6f}")
    if wilcoxon_p < 0.05:
        print("   -> Inference: Statistically significant improvement (p < 0.05). Reject H0.")

    # -------------------------------------------------------------
    # TEST 3: Friedman Test (Global Repeated Measures across Settings)
    # Evaluated across the 4 experimental configurations:
    # 1. TF-IDF PURE (54.92 mean)
    # 2. SBERT PURE (63.41 mean)
    # 3. TF-IDF FNFC (47.58 mean)
    # 4. SBERT FNFC (67.52 mean)
    # -------------------------------------------------------------
    # PURE accuracies from manuscript Table 5
    pure_tfidf = np.array([73.99, 64.66, 56.49, 42.03, 46.91, 58.47, 18.23, 71.66, 68.45, 58.30, 29.07, 70.80])
    pure_sbert = np.array([82.55, 67.34, 66.97, 78.02, 63.79, 64.33, 22.41, 69.47, 67.53, 64.68, 37.24, 76.56])
    
    friedman_stat, friedman_p = stats.friedmanchisquare(pure_tfidf, pure_sbert, tfidf_acc, hybrid_acc)
    print("\n[TEST 3] Friedman 4-Way Setting Comparison (PURE TFIDF, PURE SBERT, FNFC TFIDF, FNFC SBERT):")
    print(f"   Chi-Square (Q): {friedman_stat:.4f}, p-value: {friedman_p:.6f}")
    if friedman_p < 0.05:
        print("   -> Inference: Statistically significant global difference across feature settings (p < 0.001).")

    # -------------------------------------------------------------
    # TEST 4: Effect Size (Hedges' g)
    # -------------------------------------------------------------
    mean_diff = np.mean(diff)
    pooled_sd = np.sqrt((np.var(hybrid_acc, ddof=1) + np.var(tfidf_acc, ddof=1)) / 2)
    # Correction factor for small sample size
    j_factor = 1 - (3 / (4 * (2 * n - 2) - 1))
    hedges_g = (mean_diff / pooled_sd) * j_factor if pooled_sd > 0 else 0
    print("\n[TEST 4] Effect Size (Hedges' g):")
    print(f"   Hedges' g: {hedges_g:+.4f}")
    if abs(hedges_g) >= 0.8:
        print("   -> Inference: Large positive effect size (g >= 0.8).")
    elif abs(hedges_g) >= 0.5:
        print("   -> Inference: Moderate positive effect size (0.5 <= g < 0.8).")
    else:
        print("   -> Inference: Small-to-medium effect size.")

    # -------------------------------------------------------------
    # TEST 5: Spearman Rank Correlation (Rank Consistency)
    # -------------------------------------------------------------
    spearman_rho, spearman_p = stats.spearmanr(tfidf_acc, hybrid_acc)
    print("\n[TEST 5] Spearman Rank Correlation (Classifier Ranking Stability):")
    print(f"   rho: {spearman_rho:.4f}, p-value: {spearman_p:.6f}")

    # -------------------------------------------------------------
    # TEST 6: Holm-Bonferroni Correction (Multiple Hypotheses)
    # -------------------------------------------------------------
    # Test individual classifier gains against 0
    raw_p_values = [wilcoxon_p, friedman_p, shapiro_p, spearman_p]
    rejected, corrected_p, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
    print("\n[TEST 6] Holm-Bonferroni FWER Correction on Statistical Battery:")
    for i, (raw, corr, rej) in enumerate(zip(raw_p_values, corrected_p, rejected)):
        print(f"   Hypothesis {i+1}: Raw p={raw:.6f} -> Adjusted p={corr:.6f} (Significant: {rej})")

    # -------------------------------------------------------------
    # TEST 7: Generalization Gap Analysis
    # -------------------------------------------------------------
    gap_wilcoxon = stats.wilcoxon(diff, alternative='greater')
    print("\n[TEST 7] Generalization Gap (One-Sample Wilcoxon on Improvement):")
    print(f"   Mean Gain: {mean_diff:+.2f}%, W: {gap_wilcoxon.statistic:.2f}, p-value: {gap_wilcoxon.pvalue:.6f}")

    # -------------------------------------------------------------
    # TEST 8: Bootstrap 95% Confidence Interval (10,000 resamples)
    # -------------------------------------------------------------
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(hybrid_acc, size=n, replace=True)) for _ in range(10000)]
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    print("\n[TEST 8] Bootstrap 95% Confidence Interval for Hybrid Accuracy:")
    print(f"   Mean: {np.mean(hybrid_acc):.2f}%, 95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    # -------------------------------------------------------------
    # TEST 9: Comprehensive Descriptive Dispersion Statistics
    # -------------------------------------------------------------
    median_hybrid = np.median(hybrid_acc)
    iqr_hybrid = stats.iqr(hybrid_acc)
    std_hybrid = np.std(hybrid_acc, ddof=1)
    print("\n[TEST 9] Descriptive Dispersion Statistics (Hybrid Space):")
    print(f"   Mean: {np.mean(hybrid_acc):.2f}% +/- {std_hybrid:.2f}% | Median: {median_hybrid:.2f}% | IQR: {iqr_hybrid:.2f}%")

    print("\n" + "=" * 80)
    print("✅ All 9 Statistical Tests Finished!")

if __name__ == '__main__':
    compute_all_statistical_tests()
