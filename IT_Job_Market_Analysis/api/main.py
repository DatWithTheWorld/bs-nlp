from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crawler.app_reviews_scraper import scrape_single_app
from nlp.sentiment_analyzer import get_sentiment, auto_train_lexicon

app = FastAPI(title="Live Business Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), "..", "data", "processed", filename)

def get_master_data_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "raw", "master_training_data.csv")

@app.get("/")
def read_root(): return {"message": "Welcome to Live Sentiment API"}

@app.get("/reviews")
def get_reviews():
    try:
        df = pd.read_csv(get_data_path("processed_reviews.csv"))
        df = df.replace({float('nan'): None})
        return df.head(1000).to_dict(orient="records")
    except:
        return []

@app.get("/sentiment/summary")
def get_summary():
    try:
        with open(get_data_path("sentiment_results.json"), "r") as f:
            return json.load(f)
    except:
        return {}

@app.get("/analyze-live")
def analyze_live(app_id: str, background_tasks: BackgroundTasks, limit: int = 150):
    scraped_reviews = scrape_single_app(app_id, count=limit)
    if not scraped_reviews:
        return {"error": f"Could not fetch reviews for App ID '{app_id}'. Please ensure the ID is correct."}
        
    all_tokens_pos = []
    all_tokens_neg = []
        
    for r in scraped_reviews:
        sentiment, polarity, final_features = get_sentiment(r['review_text'])
        r['predicted_sentiment'] = sentiment
        r['polarity_score'] = polarity
        r['cleaned_text'] = ", ".join([f[0] for f in final_features]) if final_features else ""
        
        # Term Frequency Logging for N-grams Extraction
        for feature, f_type in final_features:
            if f_type == "Positive":
                all_tokens_pos.append(feature)
            elif f_type == "Negative":
                all_tokens_neg.append(feature)
            
    df = pd.DataFrame(scraped_reviews)
    
    # ML Continuous Learning Data Logging
    master_path = get_master_data_path()
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    if os.path.exists(master_path):
        df.to_csv(master_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(master_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        
    # Trigger AI Vocabulary Expansion safely in Background without blocking API response
    background_tasks.add_task(auto_train_lexicon, master_path)
    
    # Update Global Dashboard Data Aggregation
    processed_path = get_data_path("processed_reviews.csv")
    try:
        if os.path.exists(processed_path):
            global_df = pd.read_csv(processed_path)
            global_df = global_df[global_df['company'] != app_id] # Anti-duplicate
            global_df = pd.concat([global_df, df], ignore_index=True)
        else:
            global_df = df
        global_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
        
        # Re-calc Global metrics
        global_dist = {str(k): int(v) for k, v in global_df['predicted_sentiment'].value_counts().to_dict().items()}
        global_polarity = global_df.groupby('company')['polarity_score'].mean().to_dict()
        with open(get_data_path("sentiment_results.json"), "w") as f:
            json.dump({
                "sentiment_distribution": global_dist,
                "company_polarity": global_polarity
            }, f, indent=4)
    except Exception as e:
        print("Could not update global DB:", e)
        
    distribution = {str(k): int(v) for k, v in df['predicted_sentiment'].value_counts().to_dict().items()}
    avg_polarity = float(df['polarity_score'].mean())
    
    # Feature Extraction: Top Positive Highlights
    top_pos_words = [{"word": k, "count": v} for k, v in Counter(all_tokens_pos).most_common(10) if len(k) >= 2]
    
    # NLP Bug Priority Ranking
    SEVERITY_WEIGHTS = {
        "lừa": 5, "nuốt": 5, "trừ": 5, "mất": 5, "hack": 5, "đảo": 5,
        "lỗi": 4, "văng": 4, "treo": 4, "đơ": 4, "xóa": 4, "mật": 4, "đăng nhập": 4, "không được": 4, "vào được": 4,
        "chậm": 2, "lag": 2, "khó": 2, "cập nhật": 2, "rác": 3,
    }
    
    neg_counter = Counter(all_tokens_neg)
    prioritized_bugs = []
    for phrase, count in neg_counter.items():
        if len(phrase) >= 2:
            weight = 1
            for sev_word, w in SEVERITY_WEIGHTS.items():
                if sev_word in phrase:
                    weight = max(weight, w)
                    
            priority_score = count * weight
            if weight >= 5: level = "[CRITICAL] 🚨"
            elif weight >= 4: level = "[HIGH] 🔴"
            elif weight >= 2: level = "[MEDIUM] 🟡"
            else: level = "[LOW] ⚪"
            prioritized_bugs.append({"raw_word": phrase, "word": f"{phrase} {level}", "count": priority_score})
            
    # Sort by calculated priority score and pick top 10
    prioritized_bugs.sort(key=lambda x: x['count'], reverse=True)
    top_neg_words = prioritized_bugs[:10]
    
    # Time-series Extraction (Average Rating per day)
    df_date = df.copy()
    try:
        df_date['review_date'] = pd.to_datetime(df_date['review_date'])
        time_series = df_date.groupby(df_date['review_date'].dt.strftime('%m-%d'))['true_score'].mean().reset_index()
        time_series_data = time_series.rename(columns={"review_date": "date", "true_score": "average_score"}).to_dict(orient="records")
    except:
        time_series_data = []
    
    return {
        "reviews": scraped_reviews,
        "summary": {
            "sentiment_distribution": distribution,
            "company_polarity": {app_id: avg_polarity},
            "company_breakdown": {app_id: distribution},
            "total_reviews": len(df)
        },
        "nlp_insights": {
            "top_positive_words": top_pos_words,
            "top_negative_words": top_neg_words,
            "time_series_trend": time_series_data
        }
    }
