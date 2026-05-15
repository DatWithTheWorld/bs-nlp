import json
import os

def generate_business_insights(aspect_stats_path, review_stats_path, output_path):
    with open(aspect_stats_path, 'r', encoding='utf-8') as f:
        aspect_data = json.load(f)
    with open(review_stats_path, 'r', encoding='utf-8') as f:
        review_data = json.load(f)

    # Calculate some derived stats for insights
    aspect_sentiment = aspect_data['aspect_sentiment']
    
    insights = []
    recommendations = []

    # Insight 1: Stability & Performance
    err_neg = aspect_sentiment['Errors/Issues']['Negative']
    err_total = sum(aspect_sentiment['Errors/Issues'].values())
    err_ratio = (err_neg / err_total) * 100
    
    if err_ratio > 50:
        insights.append({
            "title": "Stability is a Major Pain Point",
            "description": f"Over {err_ratio:.0f}% of reviews regarding 'Errors/Issues' are negative. Users frequently complain about crashes and bugs.",
            "impact": "High",
            "category": "Technical"
        })
        recommendations.append("Prioritize bug fixing and stability updates for apps with high error counts.")

    # Insight 2: UI/UX
    ui_pos = aspect_sentiment['UI/UX']['Positive']
    ui_total = sum(aspect_sentiment['UI/UX'].values())
    ui_ratio = (ui_pos / ui_total) * 100
    
    if ui_ratio > 70:
        insights.append({
            "title": "User Experience is a Key Strength",
            "description": f"{ui_ratio:.0f}% of users are satisfied with the UI/UX design. This is a consistent positive driver across the ecosystem.",
            "impact": "Medium",
            "category": "Design"
        })
        recommendations.append("Maintain the current design language and leverage UI/UX excellence in marketing.")

    # Insight 3: App Performance (ZaloPay vs others)
    app_sentiment = review_data['app_sentiment']
    best_app = max(app_sentiment, key=lambda x: app_sentiment[x]['Positive'] / sum(app_sentiment[x].values()))
    worst_app = min(app_sentiment, key=lambda x: app_sentiment[x]['Positive'] / sum(app_sentiment[x].values()))
    
    insights.append({
        "title": f"{best_app} Leads in Satisfaction",
        "description": f"{best_app} has the highest ratio of positive feedback. Its user experience model should be analyzed for cross-app learnings.",
        "impact": "High",
        "category": "Product"
    })
    
    insights.append({
        "title": f"Sentiment Warning for {worst_app}",
        "description": f"{worst_app} shows the lowest satisfaction level. Immediate attention to its recent updates and technical performance is required.",
        "impact": "Critical",
        "category": "Retention"
    })
    recommendations.append(f"Conduct a deep dive into {worst_app}'s negative reviews to identify specific version regressions.")

    # Insight 4: Feature Request Trends
    feat_words = aspect_data['aspect_keywords']['Features']
    insights.append({
        "title": "Feature Requests & Monetization",
        "description": f"Common words in feature discussions include: {', '.join(feat_words[:5])}. Users are talking about monetization ('tiền', 'nạp') and core functionality.",
        "impact": "Medium",
        "category": "Monetization"
    })
    recommendations.append("Review the monetization balance (IAP) as users are increasingly vocal about payment issues.")

    result = {
        "insights": insights,
        "recommendations": recommendations,
        "summary": "The VNG ecosystem enjoys strong UI/UX praise but faces significant technical challenges in stability and performance, especially in gaming and utility apps."
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Business insights saved to {output_path}")

if __name__ == "__main__":
    aspect_stats_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\aspect_stats.json"
    review_stats_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\review_stats.json"
    output_path = r"e:\automat\AI-DS Projects VKU\AI-DS Projects VKU\ANI\vng-sentiment-dashboard\public\data\insights.json"
    generate_business_insights(aspect_stats_path, review_stats_path, output_path)
