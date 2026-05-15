import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import sys

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def discover_topics_in_others(csv_path, n_topics=8):
    df = pd.read_csv(csv_path)
    
    # Process full dataset instead of just 'Others'
    work_df = df.copy()
    
    if len(work_df) < 100:
        print("Not enough data in 'Others' for topic modeling.")
        return

    # Use TF-IDF to vectorize text
    tfidf = TfidfVectorizer(max_df=0.9, min_df=5, stop_words=None) # Stop words already handled in preprocessing
    dtm = tfidf.fit_transform(work_df['processed_text'].astype(str))
    
    # LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    # Display results
    print(f"\nDiscovered {n_topics} latent topics in 'Others' category:")
    feature_names = tfidf.get_feature_names_out()
    
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        print(f"Topic #{i+1}: {', '.join(top_words)}")

if __name__ == "__main__":
    csv_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng_sentiment_analysis\output\processed_reviews.csv"
    discover_topics_in_others(csv_path)
