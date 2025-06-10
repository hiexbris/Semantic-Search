import requests # type: ignore
from datetime import datetime
import time
import json

# Parameters
START_YEAR = 2019
START_MONTH = 6  # June 2019, roughly 5 years ago (adjust if you want exactly 5 years from now)
END_YEAR = datetime.now().year
END_MONTH = datetime.now().month
TOP_K = 25000

# Filter function for removing unwanted titles
def is_valid_title(title):
    invalid_prefixes = [
        "Main_Page", "Special:", "Talk:", "User:", "User_talk:", "Wikipedia:",
        "File:", "MediaWiki:", "Template:", "Help:", "Category:", "Portal:",
        "Draft:", "TimedText:", "Module:", "Gadget:", "Gadget_talk:"
    ]
    for prefix in invalid_prefixes:
        if title.startswith(prefix):
            return False
    # Also ignore empty titles or any weird chars (optional)
    if not title or title.strip() == "":
        return False
    return True

# Generate (year, month) tuples from start to end
def generate_month_year_pairs(start_year, start_month, end_year, end_month):
    pairs = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        pairs.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pairs

def fetch_monthly_top_pages(year, month):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{str(month).zfill(2)}/01"
    headers = {
        "User-Agent": "hiexbris/1.0 (hellojiaditya@gmail.com)"  # polite user-agent
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["items"][0]["articles"]  # list of dicts with "article" and "views"
        else:
            print(f"Warning: Failed to fetch {year}-{month}: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception for {year}-{month}: {e}")
        return []

def main():
    # Step 1: Generate months to query
    months_to_query = generate_month_year_pairs(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    print(f"Fetching top pages for {len(months_to_query)} months...")

    # Step 2: Accumulate view counts
    view_counts = {}

    for (year, month) in months_to_query:
        print(f"Fetching {year}-{str(month).zfill(2)} ...")
        articles = fetch_monthly_top_pages(year, month)
        for entry in articles:
            title = entry["article"]
            views = entry["views"]
            if not is_valid_title(title):
                continue
            # Accumulate counts
            if title not in view_counts:
                view_counts[title] = 0
            view_counts[title] += views
        
        # Respect API usage limits — add delay (e.g. 1-2 seconds)
        time.sleep(1.5)

    print(f"Total unique valid articles collected: {len(view_counts)}")

    # Step 3: Sort articles by total views descending
    sorted_articles = sorted(view_counts.items(), key=lambda x: x[1], reverse=True)

    top_articles = sorted_articles[:TOP_K]

    # Step 5: Write titles to file

    output_filename = f"dataset/top_{TOP_K}_wikipedia_titles.json"
    data = [{ "title": title, "views": views } for title, views in top_articles]
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Top {TOP_K} titles saved to {output_filename}")

if __name__ == "__main__":
    main()
