import os
import argparse
import pandas as pd
from src.features import normalize_promise_label, PROMISE_CLASSES, CLASS_DESCRIPTIONS, DNAFeatureExtractor
from src.models import get_paper_classifiers
from src.evaluation import BenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser(description="DNA-Inspired Software Requirements Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE dataset")
    parser.add_argument("--demo", action="store_true", help="Run 1 split demo mode for fast verification")
    parser.add_argument("--no_sequence_model", action="store_true", help="Exclude proposed BiLSTM-Attention Hybrid Sequence Model from benchmark")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save classification results")
    args = parser.parse_args()

    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("🧬 DNA-Inspired Requirements Classification Pipeline")
    print("=" * 65)
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}!")
        
    df = pd.read_csv(args.data_path)
    df['Type'] = df['Type'].str.strip()
    df['Canonical_Class'] = df['Type'].apply(normalize_promise_label)
    
    X_text = df['Requirement'].tolist()
    y = df['Canonical_Class'].values
    
    print(f"📊 Dataset loaded successfully: {len(X_text)} requirements.")
    print(f"📋 Canonical 12-Class PROMISE Distribution (Zero 'N' class):")
    for cls in PROMISE_CLASSES:
        cnt = (y == cls).sum()
        desc = CLASS_DESCRIPTIONS.get(cls, cls)
        print(f"   {cls:2s} ({desc:16s}): {cnt:3d} ({cnt/len(y)*100:4.1f}%)")
    print("-" * 65)
    
    # 2. Extract SBERT Dense Embeddings (Inference-only, label-agnostic)
    print("🧬 Pre-computing SBERT Semantic Embeddings (all-MiniLM-L6-v2)...")
    extractor = DNAFeatureExtractor(include_taxonomic_dna=True)
    sbert_embeddings = extractor.encode_sbert(X_text)
    print(f"✅ SBERT Dense Representation: {sbert_embeddings.shape}")
    print(f"✅ Full Hybrid Feature Space: 384 (SBERT) + 50 (Fold-pure TF-IDF) + 28 (DNA Linguistic Rules) = 462 Dimensions")
    print("-" * 65)

    # 3. Initialize Models
    include_seq = not args.no_sequence_model
    classifiers = get_paper_classifiers(include_proposed_sequence_model=include_seq)
    if include_seq:
        print("🧠 Proposed Flagship Model: ACTIVE (PyTorch BiLSTM + Self-Attention)")
    print(f"🏋️ Benchmarking {len(classifiers)} Machine Learning Classifiers...")
    print("-" * 65)

    # 4. Run Benchmark with Strict Fold-Pure Isolation (Zero Data Leakage)
    evaluator = BenchmarkEvaluator(classifiers)
    num_splits = 1 if args.demo else 30
    
    summary_df = evaluator.run_benchmark(
        texts=X_text,
        y=y,
        sbert_embeddings=sbert_embeddings,
        extractor=extractor,
        num_splits=num_splits
    )
    
    print("\n" + "=" * 65)
    print("🏆 FINAL CLASSIFICATION BENCHMARK SUMMARY (12 CLASSES, 462-D)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)

    # 5. Save results
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "classification_performance.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Results saved successfully to {summary_path}")

    # 6. Output LaTeX Code for the paper
    print("\n📋 LaTeX Table Source Code for Paper:")
    print("-" * 65)
    print(summary_df.to_latex(index=False, caption="10-Fold CV Performance on 12-Class PROMISE (462-D Hybrid Feature Space)", label="tab:promise_main_results"))
    print("-" * 65)

if __name__ == "__main__":
    main()
