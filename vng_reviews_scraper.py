"""
VNG Apps Review Scraper
Scrape reviews from Google Play Store for VNG Corporation apps.
Output: CSV + JSON files in vng_reviews_data/ folder.
"""

import json
import time
import os
from datetime import datetime

import pandas as pd
from google_play_scraper import Sort, reviews as gplay_reviews, app as gplay_app

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vng_reviews_data")
MAX_REVIEWS_PER_APP = 3000  # Max reviews to fetch per app
SLEEP_BETWEEN_APPS = 3      # seconds between apps to avoid rate limiting
BATCH_SIZE = 200             # reviews per batch request

# 10 VNG Apps - Google Play package names
VNG_APPS = {
    "Zalo": "com.zing.zalo",
    "Zing MP3": "com.zing.mp3",
    "ZaloPay": "vn.com.vng.zalopay",
    "Báo Mới": "vn.com.baomoi",
    "PUBG Mobile VN": "com.vng.pubgmobile",
    "Play Together VN": "com.haegin.playtogether.vng",
    "Dead Target": "com.vng.g6.a.zombie",
    "ZingSpeed Mobile": "com.vng.speedvn",
    "Thiên Long Bát Bộ 2 VNG": "com.vng.tlbb2",
    "Võ Lâm Truyền Kỳ Mobile": "com.vng.vltk.mobile",
}


def scrape_app_reviews(app_name, package_id):
    """Scrape reviews for a single app from Google Play Store."""
    print(f"\n{'─' * 50}")
    print(f"🔄 Scraping: {app_name} ({package_id})")
    print(f"{'─' * 50}")

    app_info_data = {}
    reviews_list = []

    # Step 1: Get app info
    try:
        info = gplay_app(package_id, lang='vi', country='vn')
        app_info_data = {
            "title": info.get("title", app_name),
            "score": info.get("score", "N/A"),
            "ratings": info.get("ratings", "N/A"),
            "reviews_count": info.get("reviews", "N/A"),
            "installs": info.get("installs", "N/A"),
            "developer": info.get("developer", "N/A"),
            "genre": info.get("genre", "N/A"),
        }
        print(f"   📱 {app_info_data['title']}")
        print(f"   ⭐ Rating: {app_info_data['score']} | Reviews: {app_info_data['reviews_count']} | Installs: {app_info_data['installs']}")
    except Exception as e:
        print(f"   ⚠️ Could not get app info: {e}")

    # Step 2: Fetch reviews in batches
    continuation_token = None
    fetched = 0
    retries = 0
    max_retries = 3

    while fetched < MAX_REVIEWS_PER_APP:
        try:
            batch, continuation_token = gplay_reviews(
                package_id,
                lang='vi',
                country='vn',
                sort=Sort.NEWEST,
                count=min(BATCH_SIZE, MAX_REVIEWS_PER_APP - fetched),
                continuation_token=continuation_token,
            )

            if not batch:
                print(f"   📭 No more reviews available")
                break

            reviews_list.extend(batch)
            fetched += len(batch)
            print(f"   📝 Fetched: {fetched} reviews...")

            if continuation_token is None:
                print(f"   📭 Reached end of reviews")
                break

            retries = 0  # Reset retry counter on success
            time.sleep(1)  # Rate limit between batches

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"   ❌ Max retries reached. Error: {e}")
                break
            print(f"   ⚠️ Retry {retries}/{max_retries}: {e}")
            time.sleep(5)

    # Step 3: Process reviews
    processed_reviews = []
    for review in reviews_list:
        processed_reviews.append({
            "platform": "Google Play",
            "app_name": app_name,
            "package_id": package_id,
            "review_id": review.get("reviewId", ""),
            "username": review.get("userName", ""),
            "user_image": review.get("userImage", ""),
            "content": review.get("content", ""),
            "score": review.get("score", ""),
            "thumbs_up": review.get("thumbsUpCount", 0),
            "review_date": str(review.get("at", "")),
            "reply_content": review.get("replyContent", ""),
            "reply_date": str(review.get("repliedAt", "")),
            "app_version": review.get("reviewCreatedVersion", ""),
        })

    print(f"   ✅ Collected: {len(processed_reviews)} reviews for {app_name}")
    return processed_reviews, app_info_data


def save_reviews(reviews, filename_prefix):
    """Save reviews to CSV and JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as CSV (with UTF-8 BOM for Excel compatibility)
    csv_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{timestamp}.csv")
    df = pd.DataFrame(reviews)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved CSV: {csv_path} ({len(reviews)} reviews)")

    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 Saved JSON: {json_path}")

    return csv_path, json_path


def save_per_app(reviews, app_name):
    """Save reviews for a single app to its own CSV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = app_name.replace(" ", "_").replace(":", "").lower()
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_reviews.csv")
    df = pd.DataFrame(reviews)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def main():
    print("🚀 VNG Apps Review Scraper - Google Play Store")
    print(f"📦 Output directory: {OUTPUT_DIR}")
    print(f"📱 Apps to scrape: {len(VNG_APPS)}")
    print(f"📝 Max reviews per app: {MAX_REVIEWS_PER_APP}")
    print(f"⏱️ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_reviews = []
    app_summaries = {}
    failed_apps = []

    for app_name, package_id in VNG_APPS.items():
        try:
            reviews, app_info = scrape_app_reviews(app_name, package_id)

            if reviews:
                all_reviews.extend(reviews)
                # Save individual app file
                per_app_path = save_per_app(reviews, app_name)
                print(f"   💾 Individual file: {per_app_path}")

            app_summaries[app_name] = {
                "package_id": package_id,
                "app_info": app_info,
                "reviews_scraped": len(reviews),
            }

        except Exception as e:
            print(f"   ❌ Failed to scrape {app_name}: {e}")
            failed_apps.append({"app": app_name, "package_id": package_id, "error": str(e)})

        time.sleep(SLEEP_BETWEEN_APPS)

    # Save combined file
    if all_reviews:
        print(f"\n{'=' * 70}")
        print(f"💾 SAVING COMBINED DATA")
        print(f"{'=' * 70}")
        save_reviews(all_reviews, "all_vng_reviews")

    # Generate summary report
    summary = {
        "scrape_date": datetime.now().isoformat(),
        "total_reviews": len(all_reviews),
        "total_apps_scraped": len(app_summaries),
        "failed_apps": failed_apps,
        "app_details": app_summaries,
        "reviews_per_app": {
            name: data["reviews_scraped"] for name, data in app_summaries.items()
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "scrape_summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Print final summary
    print(f"\n{'=' * 70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"📅 Date: {summary['scrape_date']}")
    print(f"📱 Apps scraped: {summary['total_apps_scraped']}/{len(VNG_APPS)}")
    print(f"📝 Total reviews: {summary['total_reviews']}")
    print()
    print(f"{'App Name':<30} {'Reviews':>10}")
    print(f"{'─' * 42}")
    for name, count in summary["reviews_per_app"].items():
        print(f"{name:<30} {count:>10}")
    print(f"{'─' * 42}")
    print(f"{'TOTAL':<30} {summary['total_reviews']:>10}")

    if failed_apps:
        print(f"\n⚠️ Failed apps ({len(failed_apps)}):")
        for fa in failed_apps:
            print(f"   - {fa['app']}: {fa['error']}")

    print(f"\n📁 All files saved in: {OUTPUT_DIR}")
    print(f"⏱️ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
