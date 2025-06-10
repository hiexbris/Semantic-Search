import requests  # type: ignore

class GoogleCSERetriever:
    def __init__(self, api_key, search_engine_id, top_k):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.banned_domains = ["wikipedia.org", "wikimedia.org", "wiki"]
        self.top_k = top_k

    def _is_valid_url(self, url):
        url = url.lower()
        return not any(bad in url for bad in self.banned_domains)

    def get_top_urls(self, query):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": self.top_k
        }

        try:
            res = requests.get(url, params=params)
            results = res.json().get("items", [])
            # filtered_urls = [item["link"] for item in results if self._is_valid_url(item["link"])] # THIS IS WHEN TO IGNORE THE WIKI PAGES 
            filtered_urls = [item["link"] for item in results] # THIS IS TO INCLUDE WIKI PAGES AS WELL USING SEACH ENGINE
            return {
                "query": query,
                "top_results": filtered_urls
            }
        except Exception as e:
            print(f"Error: {e}")
            return {
                "query": query,
                "top_results": []
            }
    
    def cleanup(self):
        self.api_key = None
        self.search_engine_id = None
        self.banned_domains = None
        self.top_k = None