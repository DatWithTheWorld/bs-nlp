import pandas as pd
import json
import os

def extract_aspect_stats(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Mapping Vietnamese aspects to English
    aspect_map = {
        "Khác": "Others",
        "Lỗi/Sự cố": "Errors/Issues",
        "Hiệu năng": "Performance",
        "Tính năng": "Features",
        "Giao diện": "UI/UX"
    }
    
    df['aspect_en'] = df['aspect'].map(lambda x: aspect_map.get(x, x))
    
    # Sentiment distribution per aspect
    # sentiment mapping: 0: Negative, 1: Neutral, 2: Positive
    sent_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df['sentiment_name'] = df['sentiment'].map(sent_map)
    
    aspect_sentiment = pd.crosstab(df['aspect_en'], df['sentiment_name']).to_dict(orient='index')
    
    # Top words per aspect
    aspect_keywords = {}
    for aspect in df['aspect_en'].unique():
        subset = df[df['aspect_en'] == aspect]
        # Basic keyword extraction (top words in processed_text)
        all_words = ' '.join(subset['processed_text'].astype(str)).split()
        top_words = pd.Series(all_words).value_counts().head(10).index.tolist()
        aspect_keywords[aspect] = top_words
        
    result = {
        "aspect_sentiment": aspect_sentiment,
        "aspect_keywords": aspect_keywords
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Aspect stats saved to {output_path}")

if __name__ == "__main__":
    csv_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng_sentiment_analysis\output\processed_reviews.csv"
    output_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\aspect_stats.json"
    extract_aspect_stats(csv_path, output_path)
