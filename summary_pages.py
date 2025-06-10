import requests  # type: ignore
import time
import json
import sys, io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

TOP_K = 25000
INPUT_FILE = f"dataset/top_{TOP_K}_wikipedia_titles.json"
OUTPUT_FILE = f"dataset/top_{TOP_K}_wiki_summaries.json"

def clean_summary(text):
    # Remove patterns like [1], [citation needed], etc.
    return re.sub(r'\[\d+]|(\[citation needed\])', '', text).strip()

def fetch_intro(title):
    api_title = title.replace(" ", "_")
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        "&format=json"
        "&prop=extracts"
        "&exintro=true"
        "&explaintext=true"
        f"&titles={api_title}"
    )
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
            return ""
        else:
            print(f"[WARN] Failed to get summary for '{title}': HTTP {response.status_code}")
            return ""
    except Exception as e:
        print(f"[ERROR] Exception for '{title}': {e}")
        return ""

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        titles_data = json.load(f)  # list of dicts with title and views

    results = {}
    for i, item in enumerate(titles_data):
        title = item["title"]
        views = item["views"]
        print(f"Fetching summary {i+1}/{len(titles_data)}: {title}")
        summary = clean_summary(fetch_intro(title))
        results[title] = {
            "summary": summary,
            "views": views
        }
        time.sleep(0.1)  # polite delay

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"Saved summaries + views for {len(results)} pages to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
