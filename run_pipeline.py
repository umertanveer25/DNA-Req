import os
import argparse
import pandas as pd
from src.features import dna_mapping, DNAFeatureExtractor
from src.models import get_paper_classifiers
from src.evaluation import BenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser(description="DNA-Inspired Software Requirements Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="data/Promise_Dataset.csv", help="Path to PROMISE dataset")
    parser.add_argument("--demo", action="store_true", help="Run 1 split demo mode for fast verification")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save classification results")
    args = parser.parse_args()

    print("🧬 DNA-Inspired Requirements Classification Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}!")
        
    df = pd.read_csv(args.data_path)
    df['Type'] = df['Type'].str.strip()
    df['DNA_Target'] = df['Type'].apply(dna_mapping)
    
    X_text = df['Requirement'].tolist()
    y = df['DNA_Target'].values
    
    print(f"📊 Dataset loaded successfully: {len(X_text)} requirements.")
    print("📋 Mapped DNA Bases Distribution:")
    print(df['DNA_Target'].value_counts())
    print("-" * 60)
    
    # 2. Extract Features (DNA Fusion: TF-IDF + SBERT)
    print("🧬 Extracting DNA Hybrid Features (TF-IDF + SBERT)...")
    extractor = DNAFeatureExtractor()
    X_hybrid = extractor.fit_transform(X_text)
    print(f"✅ DNA Hybrid Feature Vector Dimension: {X_hybrid.shape[1]}")
    print("-" * 60)

    # 3. Initialize Models
    classifiers = get_paper_classifiers()

    # 4. Run Benchmark
    evaluator = BenchmarkEvaluator(classifiers)
    num_splits = 1 if args.demo else 30
    
    summary_df = evaluator.run_benchmark(X_hybrid, y, num_splits=num_splits)
    
    print("\n" + "=" * 60)
    print("🏆 FINAL CLASSIFICATION BENCHMARK SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    # 5. Save results
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "classification_performance.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Results saved successfully to {summary_path}")

    # 6. Output LaTeX Code for the paper
    print("\n📋 LaTeX Table Source Code for Paper:")
    print("-" * 60)
    print(summary_df.to_latex(index=False, caption="Performance Comparison of DNA-Inspired Feature Extraction", label="tab:results"))
    print("-" * 60)

if __name__ == "__main__":
    main()
