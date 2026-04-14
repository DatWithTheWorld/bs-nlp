"""
VNG Apps Review Scraper - Supplementary
Scrape reviews for VNG apps that were not found in the first run.
Replaces: Báo Mới, Play Together VN, Thiên Long Bát Bộ 2 VNG, Võ Lâm Truyền Kỳ Mobile
"""

import json
import time
import os
from datetime import datetime

import pandas as pd
from google_play_scraper import Sort, reviews as gplay_reviews, app as gplay_app

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vng_reviews_data")
MAX_REVIEWS_PER_APP = 3000
BATCH_SIZE = 200

# Replacement apps (correct package names from Play Store)
REPLACEMENT_APPS = {
    "Play Together VNG": "com.vng.playtogether",
    "Laban Key": "com.vng.inputmethod.labankey",
    "Thiên Long Bát Bộ VNG": "vng.games.thienlong.tlbb3d.kiemhiep",
    "Roblox VN": "com.roblox.client.vnggames",
}


def scrape_app_reviews(app_name, package_id):
    """Scrape reviews for a single app."""
    print(f"\n{'─' * 50}")
    print(f"Scraping: {app_name} ({package_id})")
    print(f"{'─' * 50}")

    app_info_data = {}
    reviews_list = []

    try:
        info = gplay_app(package_id, lang='vi', country='vn')
        app_info_data = {
            "title": info.get("title", app_name),
            "score": info.get("score", "N/A"),
            "ratings": info.get("ratings", "N/A"),
            "reviews_count": info.get("reviews", "N/A"),
            "installs": info.get("installs", "N/A"),
        }
        print(f"   App: {app_info_data['title']}")
        print(f"   Rating: {app_info_data['score']} | Reviews: {app_info_data['reviews_count']} | Installs: {app_info_data['installs']}")
    except Exception as e:
        print(f"   WARNING: Could not get app info: {e}")
        return [], app_info_data

    continuation_token = None
    fetched = 0

    while fetched < MAX_REVIEWS_PER_APP:
        try:
            batch, continuation_token = gplay_reviews(
                package_id, lang='vi', country='vn',
                sort=Sort.NEWEST,
                count=min(BATCH_SIZE, MAX_REVIEWS_PER_APP - fetched),
                continuation_token=continuation_token,
            )
            if not batch:
                break
            reviews_list.extend(batch)
            fetched += len(batch)
            print(f"   Fetched: {fetched} reviews...")
            if continuation_token is None:
                break
            time.sleep(1)
        except Exception as e:
            print(f"   Error: {e}")
            break

    processed = []
    for review in reviews_list:
        processed.append({
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

    print(f"   DONE: {len(processed)} reviews for {app_name}")
    return processed, app_info_data


def main():
    print("VNG Apps Review Scraper - Supplementary Run")
    print(f"Output: {OUTPUT_DIR}")

    all_new_reviews = []

    for app_name, package_id in REPLACEMENT_APPS.items():
        reviews, _ = scrape_app_reviews(app_name, package_id)
        if reviews:
            all_new_reviews.extend(reviews)
            # Save individual app
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            safe_name = app_name.replace(" ", "_").replace(":", "").lower()
            csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_reviews.csv")
            pd.DataFrame(reviews).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"   Saved: {csv_path}")
        time.sleep(3)

    # Save supplementary combined
    if all_new_reviews:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(OUTPUT_DIR, f"supplementary_reviews_{timestamp}.csv")
        pd.DataFrame(all_new_reviews).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved supplementary CSV: {csv_path} ({len(all_new_reviews)} reviews)")

        json_path = os.path.join(OUTPUT_DIR, f"supplementary_reviews_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_new_reviews, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved supplementary JSON: {json_path}")

    # Now merge with existing data
    print("\n--- Merging all data ---")
    existing_csv = None
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("all_vng_reviews_") and f.endswith(".csv"):
            existing_csv = os.path.join(OUTPUT_DIR, f)
            break

    if existing_csv and all_new_reviews:
        existing_df = pd.read_csv(existing_csv, encoding="utf-8-sig")
        new_df = pd.DataFrame(all_new_reviews)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Save final combined
        final_csv = os.path.join(OUTPUT_DIR, f"all_vng_reviews_final.csv")
        combined_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
        print(f"Final combined CSV: {final_csv} ({len(combined_df)} reviews)")

        final_json = os.path.join(OUTPUT_DIR, f"all_vng_reviews_final.json")
        combined_df.to_json(final_json, orient="records", force_ascii=False, indent=2)
        print(f"Final combined JSON: {final_json}")

        # Print summary
        print(f"\n{'=' * 50}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 50}")
        app_counts = combined_df.groupby('app_name').size()
        for app, count in app_counts.items():
            print(f"  {app}: {count} reviews")
        print(f"  ---")
        print(f"  TOTAL: {len(combined_df)} reviews")
    elif all_new_reviews:
        final_csv = os.path.join(OUTPUT_DIR, f"all_vng_reviews_final.csv")
        pd.DataFrame(all_new_reviews).to_csv(final_csv, index=False, encoding="utf-8-sig")
        print(f"Final CSV: {final_csv} ({len(all_new_reviews)} reviews)")

    print("\nDone!")


if __name__ == "__main__":
    main()
