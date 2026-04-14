"""
Data Preprocessing for VNG App Reviews Sentiment Analysis
- Load and clean review data
- Vietnamese text preprocessing
- Create sentiment labels from star ratings
- Train/test split
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Try to import Vietnamese NLP tools
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("[WARNING] underthesea not available. Using basic tokenization.")


# Vietnamese stopwords (common list)
VIETNAMESE_STOPWORDS = set([
    'và', 'của', 'có', 'là', 'cho', 'với', 'các', 'được', 'trong', 'không',
    'này', 'một', 'những', 'để', 'đã', 'nhưng', 'từ', 'khi', 'cũng', 'theo',
    'đến', 'về', 'như', 'hay', 'hoặc', 'tại', 'bị', 'nên', 'rất', 'thì',
    'đó', 'vì', 'mà', 'do', 'nó', 'ra', 'lại', 'còn', 'bởi', 'nếu',
    'ở', 'lên', 'xuống', 'vào', 'trên', 'dưới', 'qua', 'sau', 'trước',
    'ai', 'gì', 'nào', 'đâu', 'bao', 'sao', 'thế', 'vậy', 'này', 'kia',
    'ạ', 'à', 'ơi', 'ừ', 'nhé', 'nha', 'hen', 'nghen', 'á', 'ha',
    'tôi', 'tao', 'mình', 'ta', 'chúng', 'bạn', 'anh', 'chị', 'em',
    'nó', 'họ', 'ông', 'bà', 'cô', 'chú', 'thầy', 'cô',
    'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for',
    'ok', 'oke', 'okey', 'okay',
])


def clean_text(text):
    """Clean and normalize Vietnamese text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove emojis and special unicode
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    # Remove numbers (standalone)
    text = re.sub(r'\b\d+\b', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text):
    """Tokenize Vietnamese text."""
    if not text:
        return ""

    if HAS_UNDERTHESEA:
        try:
            tokens = word_tokenize(text, format="text")
            return tokens
        except Exception:
            pass

    # Fallback: simple split
    return text


def remove_stopwords(text):
    """Remove Vietnamese and English stopwords."""
    if not text:
        return ""
    words = text.split()
    words = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]
    return ' '.join(words)


def create_sentiment_label(score):
    """Convert star rating to sentiment label.
    1-2 stars -> 0 (Negative)
    3 stars   -> 1 (Neutral)
    4-5 stars -> 2 (Positive)
    """
    if score <= 2:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def get_sentiment_name(label):
    """Get sentiment name from label."""
    mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return mapping.get(label, 'Unknown')


def load_and_preprocess(data_dir, output_dir):
    """Main preprocessing pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('all_vng_reviews') and f.endswith('.csv')]
    if not csv_files:
        # Try individual app files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('_reviews.csv')]

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, f), encoding='utf-8-sig')
            dfs.append(df)
            print(f"   Loaded: {f} ({len(df)} rows)")
        except Exception as e:
            print(f"   Error loading {f}: {e}")

    if not dfs:
        raise ValueError("No data files found!")

    df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on review_id
    initial_count = len(df)
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    print(f"   Total: {initial_count} -> {len(df)} (removed {initial_count - len(df)} duplicates)")

    # Step 2: Filter valid reviews
    print("\n[2/6] Filtering valid reviews...")
    df = df[df['content'].notna() & (df['content'].str.strip() != '')]
    df = df[df['score'].notna()]
    df['score'] = df['score'].astype(int)
    print(f"   Valid reviews: {len(df)}")

    # Step 3: Create labels
    print("\n[3/6] Creating sentiment labels...")
    df['sentiment'] = df['score'].apply(create_sentiment_label)
    df['sentiment_name'] = df['sentiment'].apply(get_sentiment_name)

    label_counts = df['sentiment_name'].value_counts()
    print(f"   Label distribution:")
    for label, count in label_counts.items():
        print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")

    # Step 4: Text preprocessing
    print("\n[4/6] Preprocessing text...")
    df['clean_text'] = df['content'].apply(clean_text)
    df['tokenized_text'] = df['clean_text'].apply(tokenize_text)
    df['processed_text'] = df['tokenized_text'].apply(remove_stopwords)

    # Remove empty processed texts
    df = df[df['processed_text'].str.strip() != '']
    print(f"   After text cleaning: {len(df)} reviews")

    # Add text length feature
    df['text_length'] = df['processed_text'].apply(lambda x: len(x.split()))

    # Step 5: Train/Test split
    print("\n[5/6] Splitting data (80/20)...")
    X = df['processed_text'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Train distribution: {dict(Counter(y_train))}")
    print(f"   Test distribution: {dict(Counter(y_test))}")

    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")

    # Save full processed dataframe
    df.to_csv(os.path.join(output_dir, 'processed_reviews.csv'),
              index=False, encoding='utf-8-sig')

    # Save train/test splits
    np.savez(os.path.join(output_dir, 'train_test_split.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    # Save metadata
    metadata = {
        'total_reviews': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_classes': 3,
        'class_names': ['Negative', 'Neutral', 'Positive'],
        'label_distribution': {
            'Negative': int((y == 0).sum()),
            'Neutral': int((y == 1).sum()),
            'Positive': int((y == 2).sum()),
        },
        'apps': df['app_name'].unique().tolist() if 'app_name' in df.columns else [],
        'avg_text_length': float(df['text_length'].mean()),
        'has_underthesea': HAS_UNDERTHESEA,
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n   Saved to: {output_dir}")
    print("   Files: processed_reviews.csv, train_test_split.npz, metadata.json")
    print("\nDone!")

    return df, X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE), 'vng_reviews_data')
    OUTPUT_DIR = os.path.join(BASE, 'output')
    load_and_preprocess(DATA_DIR, OUTPUT_DIR)
