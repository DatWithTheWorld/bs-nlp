"""
Main Pipeline Runner for VNG Sentiment Analysis
Runs: Data Preprocessing -> ML Training -> DL Training -> Visualizations
"""

import os
import sys
import time
import json

# Add parent dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_preprocessing import load_and_preprocess
from ml_models import train_and_evaluate_all
from visualizations import generate_all_visualizations


def main():
    start_time = time.time()

    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'vng_reviews_data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    print("=" * 70)
    print("  VNG APP REVIEWS - SENTIMENT ANALYSIS PIPELINE")
    print("  ML (NB, LR, SVM, RF) + DL (LSTM, CNN)")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # ========================================
    # Phase 1: Data Preprocessing
    # ========================================
    print("\n\n" + "#" * 70)
    print("# PHASE 1: DATA PREPROCESSING")
    print("#" * 70)

    df, X_train, X_test, y_train, y_test, metadata = load_and_preprocess(DATA_DIR, OUTPUT_DIR)

    # ========================================
    # Phase 2: Machine Learning
    # ========================================
    print("\n\n" + "#" * 70)
    print("# PHASE 2: MACHINE LEARNING MODELS")
    print("#" * 70)

    ml_results, cv_results, trained_models, vectorizer, feature_importance = \
        train_and_evaluate_all(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    # ========================================
    # Phase 3: Deep Learning
    # ========================================
    print("\n\n" + "#" * 70)
    print("# PHASE 3: DEEP LEARNING MODELS")
    print("#" * 70)

    try:
        from dl_models import train_and_evaluate_dl
        dl_results, dl_histories, dl_cv_results = train_and_evaluate_dl(
            X_train, X_test, y_train, y_test, OUTPUT_DIR,
            max_words=15000, max_len=100, epochs=20, batch_size=64
        )
    except Exception as e:
        print(f"\n[WARNING] DL training failed: {e}")
        print("Continuing with ML results only...")
        dl_results = {}

    # ========================================
    # Phase 4: Visualizations
    # ========================================
    print("\n\n" + "#" * 70)
    print("# PHASE 4: VISUALIZATIONS")
    print("#" * 70)

    generate_all_visualizations(OUTPUT_DIR)

    # ========================================
    # Final Summary
    # ========================================
    elapsed = time.time() - start_time

    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"\n  Data: {metadata['total_reviews']} reviews")
    print(f"  Train: {metadata['train_size']} | Test: {metadata['test_size']}")

    print(f"\n  ML Models ({len(ml_results)}):")
    best_ml = max(ml_results.items(), key=lambda x: x[1]['metrics']['f1_macro'])
    for name, r in ml_results.items():
        marker = " <-- BEST" if name == best_ml[0] else ""
        print(f"    {name}: F1={r['metrics']['f1_macro']:.4f}, "
              f"Acc={r['metrics']['accuracy']:.4f}{marker}")

    if dl_results:
        print(f"\n  DL Models ({len(dl_results)}):")
        for name, r in dl_results.items():
            print(f"    {name}: F1={r['metrics']['f1_macro']:.4f}, "
                  f"Acc={r['metrics']['accuracy']:.4f}")

    charts_dir = os.path.join(OUTPUT_DIR, 'charts')
    if os.path.exists(charts_dir):
        charts = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        print(f"\n  Charts generated: {len(charts)} files")
        print(f"  Charts directory: {charts_dir}")

    print(f"\n  All output saved in: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
