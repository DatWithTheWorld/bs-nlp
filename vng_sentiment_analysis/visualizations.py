"""
Visualization module for VNG Sentiment Analysis
- ML and DL charts
- Confusion matrices, ROC curves, training curves
- Cross-validation box plots
- Data distribution charts
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
COLORS = {'Negative': '#e74c3c', 'Neutral': '#f39c12', 'Positive': '#2ecc71'}
MODEL_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


def plot_data_distribution(df, output_dir):
    """Plot data analysis charts."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VNG App Reviews - Data Analysis', fontsize=18, fontweight='bold', y=1.02)

    # 1. Sentiment Distribution
    ax = axes[0, 0]
    sentiment_counts = df['sentiment_name'].value_counts()
    colors = [COLORS.get(s, '#999') for s in sentiment_counts.index]
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, sentiment_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')

    # 2. Rating Distribution
    ax = axes[0, 1]
    rating_counts = df['score'].value_counts().sort_index()
    star_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#2ecc71']
    ax.bar(rating_counts.index, rating_counts.values, color=star_colors, edgecolor='white', linewidth=1.5)
    for i, (rating, count) in enumerate(rating_counts.items()):
        ax.text(rating, count + 50, f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title('Star Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stars')
    ax.set_ylabel('Count')
    ax.set_xticks(range(1, 6))

    # 3. Reviews per App
    ax = axes[1, 0]
    if 'app_name' in df.columns:
        app_counts = df['app_name'].value_counts()
        bars = ax.barh(app_counts.index, app_counts.values, color=MODEL_COLORS[:len(app_counts)],
                       edgecolor='white', linewidth=1)
        for bar, count in zip(bars, app_counts.values):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2.,
                    f'{count:,}', va='center', fontweight='bold', fontsize=10)
        ax.set_title('Reviews per App', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
    else:
        ax.text(0.5, 0.5, 'No app data', ha='center', va='center', transform=ax.transAxes)

    # 4. Text Length Distribution
    ax = axes[1, 1]
    if 'text_length' in df.columns:
        for sentiment in CLASS_NAMES:
            subset = df[df['sentiment_name'] == sentiment]['text_length']
            if len(subset) > 0:
                ax.hist(subset, bins=50, alpha=0.5, label=sentiment,
                        color=COLORS[sentiment], edgecolor='white')
        ax.set_title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Count')
        ax.legend(fontsize=11)
        ax.set_xlim(0, min(100, df['text_length'].quantile(0.99)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: data_distribution.png")


def plot_sentiment_by_app(df, output_dir):
    """Plot sentiment distribution per app."""
    if 'app_name' not in df.columns:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    apps = df['app_name'].unique()
    x = np.arange(len(apps))
    width = 0.25

    for i, sentiment in enumerate(CLASS_NAMES):
        counts = [len(df[(df['app_name'] == app) & (df['sentiment_name'] == sentiment)]) for app in apps]
        ax.bar(x + i * width, counts, width, label=sentiment, color=COLORS[sentiment],
               edgecolor='white', linewidth=1)

    ax.set_xlabel('App', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Sentiment Distribution per VNG App', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(apps, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_app.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: sentiment_by_app.png")


def plot_confusion_matrices(results, output_dir, prefix='ml'):
    """Plot confusion matrices for all models."""
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(results)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx] if n_models > 1 else axes[0]
        cm = np.array(result['confusion_matrix'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

    # Hide extra axes
    for idx in range(n_models, rows * cols):
        if n_models > 1:
            axes[idx].set_visible(False)

    plt.suptitle(f'Confusion Matrices ({prefix.upper()})', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrices.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {prefix}_confusion_matrices.png")


def plot_model_comparison(results, output_dir, prefix='ml'):
    """Plot model comparison bar charts."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(results.keys())
    metrics_list = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']

    fig, axes = plt.subplots(1, len(metrics_list), figsize=(6 * len(metrics_list), 6))

    for i, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        ax = axes[i]
        values = [results[m]['metrics'].get(metric, 0) for m in models]
        bars = ax.bar(range(len(models)), values, color=MODEL_COLORS[:len(models)],
                      edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=max(values), color='gray', linestyle='--', alpha=0.3)

    plt.suptitle(f'Model Comparison ({prefix.upper()})', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {prefix}_model_comparison.png")


def plot_cv_boxplot(cv_results, output_dir, prefix='ml'):
    """Plot cross-validation box plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy box plot
    ax = axes[0]
    data_acc = []
    labels = []
    for name, cv in cv_results.items():
        if 'test_accuracy' in cv:
            data_acc.append(cv['test_accuracy'])
            labels.append(name)
        elif 'accuracy' in cv:
            data_acc.append(cv['accuracy'])
            labels.append(name)

    if data_acc:
        bp = ax.boxplot(data_acc, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], MODEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    # F1 box plot
    ax = axes[1]
    data_f1 = []
    labels_f1 = []
    for name, cv in cv_results.items():
        if 'test_f1' in cv:
            data_f1.append(cv['test_f1'])
            labels_f1.append(name)
        elif 'f1_macro' in cv:
            data_f1.append(cv['f1_macro'])
            labels_f1.append(name)

    if data_f1:
        bp = ax.boxplot(data_f1, labels=labels_f1, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], MODEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Cross-Validation F1 Score (Macro)', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Cross-Validation Results ({prefix.upper()}, k=5)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_cv_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {prefix}_cv_boxplot.png")


def plot_dl_training_history(histories, output_dir):
    """Plot DL training/validation loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)

    for name, history in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax = axes[0]
        ax.plot(history['loss'], label='Train Loss', color='#3498db', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
        ax.set_title(f'{name} - Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Accuracy
        ax = axes[1]
        ax.plot(history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2)
        ax.plot(history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
        ax.set_title(f'{name} - Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.suptitle(f'{name} Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'dl_training_{safe_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: dl_training_{safe_name}.png")


def plot_all_models_comparison(ml_results, dl_results, output_dir):
    """Compare all ML and DL models together."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for name, r in ml_results.items():
        all_results[f"ML: {name}"] = r['metrics']
    for name, r in dl_results.items():
        all_results[f"DL: {name}"] = r['metrics']

    models = list(all_results.keys())
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(models))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics, ['Accuracy', 'F1', 'Precision', 'Recall'])):
        values = [all_results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=MODEL_COLORS[i], edgecolor='white', linewidth=1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('All Models Comparison (ML vs DL)', fontsize=18, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=12, loc='upper right')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, label='Baseline')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: all_models_comparison.png")


def plot_classification_reports(results, output_dir, prefix='ml'):
    """Plot detailed classification report as heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    for name, result in results.items():
        report = result.get('classification_report', {})
        if not report:
            continue

        # Extract per-class metrics
        classes = CLASS_NAMES
        metrics = ['precision', 'recall', 'f1-score']

        data = []
        for cls in classes:
            if cls in report:
                row = [report[cls].get(m, 0) for m in metrics]
            else:
                row = [0, 0, 0]
            data.append(row)

        df_report = pd.DataFrame(data, index=classes, columns=['Precision', 'Recall', 'F1-Score'])

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df_report, annot=True, fmt='.3f', cmap='YlGnBu',
                    vmin=0, vmax=1, ax=ax, cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title(f'{name} - Classification Report', fontsize=14, fontweight='bold')

        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'{prefix}_report_{safe_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Saved: {prefix}_report_*.png")


def plot_wordcloud_per_sentiment(df, output_dir):
    """Plot word clouds for each sentiment class."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("   [WARNING] wordcloud not installed. Skipping word clouds.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, sentiment in enumerate(CLASS_NAMES):
        ax = axes[i]
        text_col = 'processed_text' if 'processed_text' in df.columns else 'clean_text'
        if text_col not in df.columns:
            text_col = 'content'

        texts = df[df['sentiment_name'] == sentiment][text_col].dropna()
        all_text = ' '.join(texts.values)

        if len(all_text.strip()) < 10:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            ax.set_title(sentiment)
            continue

        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis' if sentiment == 'Neutral' else (
                'Reds' if sentiment == 'Negative' else 'Greens'),
            random_state=42,
        ).generate(all_text)

        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment} Reviews', fontsize=14, fontweight='bold',
                     color=COLORS[sentiment])

    plt.suptitle('Word Clouds by Sentiment', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wordclouds.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: wordclouds.png")


def generate_all_visualizations(output_dir):
    """Generate all visualizations from saved results."""
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Load data
    processed_csv = os.path.join(output_dir, 'processed_reviews.csv')
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv, encoding='utf-8-sig')
        print("\n[1] Data Distribution Charts...")
        plot_data_distribution(df, charts_dir)
        plot_sentiment_by_app(df, charts_dir)
        print("\n[2] Word Clouds...")
        plot_wordcloud_per_sentiment(df, charts_dir)
    else:
        print("   No processed_reviews.csv found, skipping data charts.")
        df = None

    # ML Results
    ml_results_path = os.path.join(output_dir, 'ml_results.json')
    ml_results = {}
    if os.path.exists(ml_results_path):
        with open(ml_results_path, 'r', encoding='utf-8') as f:
            ml_results = json.load(f)

        print("\n[3] ML Confusion Matrices...")
        plot_confusion_matrices(ml_results, charts_dir, prefix='ml')

        print("\n[4] ML Model Comparison...")
        plot_model_comparison(ml_results, charts_dir, prefix='ml')

        print("\n[5] ML Classification Reports...")
        plot_classification_reports(ml_results, charts_dir, prefix='ml')

    # ML CV Results
    cv_results_path = os.path.join(output_dir, 'cv_results.json')
    if os.path.exists(cv_results_path):
        with open(cv_results_path, 'r', encoding='utf-8') as f:
            cv_results = json.load(f)
        print("\n[6] ML Cross-Validation Box Plots...")
        plot_cv_boxplot(cv_results, charts_dir, prefix='ml')

    # DL Results
    dl_results_path = os.path.join(output_dir, 'dl_results.json')
    dl_results = {}
    if os.path.exists(dl_results_path):
        with open(dl_results_path, 'r', encoding='utf-8') as f:
            dl_results = json.load(f)

        print("\n[7] DL Confusion Matrices...")
        plot_confusion_matrices(dl_results, charts_dir, prefix='dl')

        print("\n[8] DL Model Comparison...")
        plot_model_comparison(dl_results, charts_dir, prefix='dl')

        print("\n[9] DL Classification Reports...")
        plot_classification_reports(dl_results, charts_dir, prefix='dl')

    # DL Training History
    dl_hist_path = os.path.join(output_dir, 'dl_histories.json')
    if os.path.exists(dl_hist_path):
        with open(dl_hist_path, 'r', encoding='utf-8') as f:
            dl_histories = json.load(f)
        print("\n[10] DL Training History Curves...")
        plot_dl_training_history(dl_histories, charts_dir)

    # DL CV Results
    dl_cv_path = os.path.join(output_dir, 'dl_cv_results.json')
    if os.path.exists(dl_cv_path):
        with open(dl_cv_path, 'r', encoding='utf-8') as f:
            dl_cv = json.load(f)
        print("\n[11] DL Cross-Validation Box Plots...")
        plot_cv_boxplot(dl_cv, charts_dir, prefix='dl')

    # All models comparison
    if ml_results and dl_results:
        print("\n[12] All Models Comparison (ML vs DL)...")
        plot_all_models_comparison(ml_results, dl_results, charts_dir)

    print(f"\n All charts saved in: {charts_dir}")
    print("Done!")


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE, 'output')
    generate_all_visualizations(OUTPUT_DIR)
