import pandas as pd
from data_preprocessing import get_aspect
import json
import os

def update_aspects_and_stats(csv_path, aspect_stats_path, metadata_path):
    print(f"Updating aspects in {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Update aspects column using the improved get_aspect function
    df['aspect'] = df['processed_text'].fillna('').apply(get_aspect)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print("CSV updated.")
    
    # Update aspect_stats.json
    print("Updating aspect_stats.json...")
    
    # Sentiment per aspect
    sent_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df['sentiment_name'] = df['sentiment'].map(sent_map)
    aspect_sentiment = pd.crosstab(df['aspect'], df['sentiment_name']).to_dict(orient='index')
    
    # Top words per aspect
    aspect_keywords = {}
    for aspect in df['aspect'].unique():
        subset = df[df['aspect'] == aspect]
        all_words = ' '.join(subset['processed_text'].astype(str)).split()
        top_words = pd.Series(all_words).value_counts().head(10).index.tolist()
        aspect_keywords[aspect] = top_words
        
    stats_result = {
        "aspect_sentiment": aspect_sentiment,
        "aspect_keywords": aspect_keywords
    }
    
    with open(aspect_stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_result, f, ensure_ascii=False, indent=2)
    
    # Update metadata.json aspect distribution
    print("Updating metadata.json...")
    aspect_dist = df['aspect'].value_counts().to_dict()
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata['aspect_distribution'] = aspect_dist
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("All stats updated.")

if __name__ == "__main__":
    csv_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng_sentiment_analysis\output\processed_reviews.csv"
    aspect_stats_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\aspect_stats.json"
    metadata_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\metadata.json"
    update_aspects_and_stats(csv_path, aspect_stats_path, metadata_path)
