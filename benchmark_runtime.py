import time
import os
import sys
import psutil
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from src.features import DNALinguisticTaxonomyExtractor
from src.preprocessor import TextPreprocessor
from src.deep_models import HybridSequenceNetwork, HybridDeepSequenceClassifier

def measure_benchmark():
    print("--- STARTING END-TO-END RUNTIME BENCHMARK ---")
    df = pd.read_csv('data/Promise_Dataset.csv')
    texts = df['Requirement'].dropna().tolist()
    print(f"Loaded {len(texts)} requirement statements from PROMISE.")

    # 1. Feature Extraction Benchmarks
    print("\n[1] Benchmarking Feature Extraction Latencies (single-instance vs batch)...")
    
    # 28-D DNA
    dna_times = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        _ = DNALinguisticTaxonomyExtractor.extract_features(t)
        dna_times.append((time.perf_counter() - t0) * 1000)
    dna_ms_single = np.mean(dna_times)
    dna_p95_single = np.percentile(dna_times, 95)
    print(f"28-D DNA Feature Extraction: Mean = {dna_ms_single:.3f} ms, P95 = {dna_p95_single:.3f} ms")

    # TF-IDF (fitting on training set)
    clean_texts = [TextPreprocessor.clean_text(t) for t in texts]
    tfidf = TfidfVectorizer(max_features=50)
    t0_fit = time.perf_counter()
    tfidf.fit(clean_texts)
    tfidf_fit_time = (time.perf_counter() - t0_fit) * 1000
    print(f"TF-IDF Vocabulary Fit Time (969 items): {tfidf_fit_time:.2f} ms")

    tfidf_times = []
    for ct in clean_texts[:200]:
        t0 = time.perf_counter()
        _ = tfidf.transform([ct])
        tfidf_times.append((time.perf_counter() - t0) * 1000)
    tfidf_ms_single = np.mean(tfidf_times)
    tfidf_p95_single = np.percentile(tfidf_times, 95)
    print(f"50-D TF-IDF Single Transform: Mean = {tfidf_ms_single:.3f} ms, P95 = {tfidf_p95_single:.3f} ms")

    # SBERT all-MiniLM-L6-v2 on CPU
    print("\nLoading SentenceTransformer('all-MiniLM-L6-v2') on CPU...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0_load = time.perf_counter()
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    sbert_load_time = (time.perf_counter() - t0_load) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)
    sbert_ram_mb = mem_after - mem_before
    print(f"SBERT Loaded in {sbert_load_time:.2f} ms. Model RAM consumption: ~{sbert_ram_mb:.1f} MB")

    # Warmup SBERT
    _ = sbert.encode(["Warmup text requirement for transformer initialization."])

    # Single-instance SBERT latency
    sbert_single_times = []
    for t in texts[:100]:
        t0 = time.perf_counter()
        _ = sbert.encode([t], show_progress_bar=False, batch_size=1)
        sbert_single_times.append((time.perf_counter() - t0) * 1000)
    sbert_ms_single = np.mean(sbert_single_times)
    sbert_p95_single = np.percentile(sbert_single_times, 95)
    print(f"SBERT Single-Sentence Encoding (CPU): Mean = {sbert_ms_single:.2f} ms, P95 = {sbert_p95_single:.2f} ms")

    # Batched SBERT latency (batch_size=32)
    t0_batch = time.perf_counter()
    _ = sbert.encode(texts[:96], batch_size=32, show_progress_bar=False)
    sbert_batch_throughput = 96 / (time.perf_counter() - t0_batch)
    sbert_batch_ms_per_item = 1000.0 / sbert_batch_throughput
    print(f"SBERT Batched Encoding (batch=32, CPU): {sbert_batch_ms_per_item:.2f} ms/item ({sbert_batch_throughput:.1f} req/s)")

    # 2. Classifier Training & Inference Latency
    print("\n[2] Benchmarking Classifiers (Random Forest vs Deep Sequence Model)...")
    
    # Prepare dummy feature matrices for timing
    X_78 = np.random.randn(len(texts), 78).astype(np.float32)
    X_462 = np.random.randn(len(texts), 462).astype(np.float32)
    y = np.random.randint(0, 12, size=len(texts))

    # Random Forest 462-D
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0_rf_train = time.perf_counter()
    rf.fit(X_462, y)
    rf_train_sec = time.perf_counter() - t0_rf_train
    print(f"Random Forest (462-D, 100 trees) Training Time: {rf_train_sec:.3f} s")

    # RF Single-Instance Inference
    rf_single_times = []
    for row in X_462[:200]:
        t0 = time.perf_counter()
        _ = rf.predict([row])
        rf_single_times.append((time.perf_counter() - t0) * 1000)
    rf_ms_single = np.mean(rf_single_times)
    rf_p95_single = np.percentile(rf_single_times, 95)
    print(f"Random Forest Single Inference: Mean = {rf_ms_single:.3f} ms, P95 = {rf_p95_single:.3f} ms")

    # RF Batched Inference (batch=969)
    t0_rf_batch = time.perf_counter()
    _ = rf.predict(X_462)
    rf_batch_throughput = len(texts) / (time.perf_counter() - t0_rf_batch)
    print(f"Random Forest Batch Throughput: {rf_batch_throughput:.1f} req/s")

    # Deep Sequence Model (BiLSTM + Attention)
    deep_model = HybridSequenceNetwork(input_dim=462, num_classes=12, hidden_dim=128)
    deep_model.eval()

    dummy_input = torch.randn(1, 462)

    # Warmup deep model
    with torch.no_grad():
        _ = deep_model(dummy_input)

    deep_single_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = deep_model(dummy_input)
        deep_single_times.append((time.perf_counter() - t0) * 1000)
    deep_ms_single = np.mean(deep_single_times)
    deep_p95_single = np.percentile(deep_single_times, 95)
    print(f"Deep Sequence Model Single Forward Pass (CPU): Mean = {deep_ms_single:.3f} ms, P95 = {deep_p95_single:.3f} ms")

    # Batched Deep Model (batch=32)
    dummy_input_32 = torch.randn(32, 462)
    t0_deep_batch = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = deep_model(dummy_input_32)
    deep_batch_throughput = 320 / (time.perf_counter() - t0_deep_batch)
    print(f"Deep Sequence Model Batch Throughput: {deep_batch_throughput:.1f} req/s")

    # 3. Summary of End-to-End Latencies
    print("\n--- SUMMARY OF END-TO-END PIPELINES ---")
    
    # Pipeline 1: Ultra-Lightweight Edge (TF-IDF 50d + DNA 28d = 78d -> Random Forest)
    e2e_ultra_edge = dna_ms_single + tfidf_ms_single + rf_ms_single
    print(f"Tier 1 (Ultra-Lightweight Edge, 78-D TF-IDF+DNA + RF): {e2e_ultra_edge:.2f} ms")
    print(f"   -> Neural Transformer Overhead: 0.00 ms (Zero deep learning models loaded)")
    print(f"   -> Accuracy on PROMISE: 63.57%")

    # Pipeline 2: Full Multimodal Edge (462-D SBERT+TF-IDF+DNA -> Random Forest)
    e2e_full_edge = sbert_ms_single + dna_ms_single + tfidf_ms_single + rf_ms_single
    print(f"Tier 2 (Full Multimodal Edge, 462-D SBERT+TFIDF+DNA + RF): {e2e_full_edge:.2f} ms")
    print(f"   -> SBERT Feature Extraction Share: {(sbert_ms_single / e2e_full_edge)*100:.1f}%")
    print(f"   -> Tree Traversal Share: {(rf_ms_single / e2e_full_edge)*100:.1f}%")
    print(f"   -> Accuracy on PROMISE: 78.67%")

    # Pipeline 3: Flagship Cloud/CI-CD (462-D Multimodal -> BiLSTM + Multi-Head Self-Attention)
    e2e_flagship = sbert_ms_single + dna_ms_single + tfidf_ms_single + deep_ms_single
    print(f"Tier 3 (Flagship CI/CD Server, 462-D + Deep Sequence Model): {e2e_flagship:.2f} ms (CPU)")
    print(f"   -> Accuracy on PROMISE: 82.77%")

    # Model parameter counts and disk footprints
    sbert_params = sum(p.numel() for p in sbert.parameters())
    deep_params = sum(p.numel() for p in deep_model.parameters())
    print(f"\nModel Parameters: SBERT = {sbert_params/1e6:.1f}M, Deep Sequence Model = {deep_params/1e3:.1f}K")

if __name__ == '__main__':
    measure_benchmark()
