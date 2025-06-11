import requests  # type: ignore

class GoogleCSERetriever:
    def __init__(self, api_key, top_k=5):
        self.api_key = api_key
        self.top_k = top_k
        self.banned_domains = ["wikipedia.org", "wikimedia.org", "wiki"]

    def _is_valid_url(self, url):
        url = url.lower()
        return not any(bad in url for bad in self.banned_domains)

    def get_top_urls(self, query):
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "num": self.top_k
        }

        try:
            response = requests.get(url, params=params)
            results = response.json().get("organic_results", [])
            # filtered_urls = [item["link"] for item in results if self._is_valid_url(item["link"])]  # To exclude Wiki
            filtered_urls = [item["link"] for item in results if "link" in item]  # Includes all
            return {
                "query": query,
                "top_results": filtered_urls[:self.top_k]
            }
        except Exception as e:
            print(f"Error: {e}")
            return {
                "query": query,
                "top_results": []
            }

    def cleanup(self):
        self.api_key = None
        self.banned_domains = None
        self.top_k = None
