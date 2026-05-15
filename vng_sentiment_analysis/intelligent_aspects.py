import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import os

def apply_intelligent_aspects(csv_path, aspect_stats_path, metadata_path):
    print("Step 1: Training LDA Model for Intelligent Topic Discovery...")
    df = pd.read_csv(csv_path)
    
    # Vectorize
    tfidf = TfidfVectorizer(max_df=0.9, min_df=5)
    dtm = tfidf.fit_transform(df['processed_text'].astype(str))
    
    # LDA with 8 topics
    n_topics = 8
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    # Define labels based on the discovered top words
    topic_labels = {
        0: "General Satisfaction",
        1: "Performance & Ads",
        2: "Connectivity & Stability",
        3: "UI/UX & Input",
        4: "App Updates",
        5: "Graphics & Quality",
        6: "Account & Security",
        7: "Monetization & Content"
    }
    
    print("Step 2: Assigning topics to each review...")
    topic_results = lda.transform(dtm)
    df['topic_idx'] = topic_results.argmax(axis=1)
    df['aspect'] = df['topic_idx'].map(topic_labels)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    print("Step 3: Refreshing Dashboard stats...")
    # Sentiment per aspect
    sent_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df['sentiment_name'] = df['sentiment'].map(sent_map)
    aspect_sentiment = pd.crosstab(df['aspect'], df['sentiment_name']).to_dict(orient='index')
    
    # Top words per aspect
    feature_names = tfidf.get_feature_names_out()
    aspect_keywords = {}
    for idx, label in topic_labels.items():
        topic_comp = lda.components_[idx]
        top_words = [feature_names[i] for i in topic_comp.argsort()[-12:]]
        aspect_keywords[label] = top_words
        
    stats_result = {
        "aspect_sentiment": aspect_sentiment,
        "aspect_keywords": aspect_keywords
    }
    
    with open(aspect_stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_result, f, ensure_ascii=False, indent=2)
    
    # Update metadata.json
    aspect_dist = df['aspect'].value_counts().to_dict()
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        metadata['aspect_distribution'] = aspect_dist
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Success: Dashboard updated with Intelligent Topic Discovery (LDA).")

if __name__ == "__main__":
    csv_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng_sentiment_analysis\output\processed_reviews.csv"
    aspect_stats_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\aspect_stats.json"
    metadata_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\metadata.json"
    apply_intelligent_aspects(csv_path, aspect_stats_path, metadata_path)
