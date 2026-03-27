from google_play_scraper import reviews, Sort
import time

def scrape_single_app(app_id, count=150):
    try:
        result, _ = reviews(
            app_id,
            lang='vi',
            country='vn',
            sort=Sort.NEWEST,
            count=count
        )
        data = []
        for r in result:
            data.append({
                "company": app_id,
                "location": "Google Play Vietnam",
                "review_text": str(r['content']).replace('\n', ' '),
                "review_date": r['at'].strftime('%Y-%m-%d'),
                "true_score": r['score']
            })
        return data
    except Exception as e:
        print(f"Error scraping {app_id}: {e}")
        return []
