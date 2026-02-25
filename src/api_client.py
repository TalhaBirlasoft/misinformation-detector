import requests
import os
from dotenv import load_dotenv

load_dotenv()

class FactCheckAPI:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def search_claim(self, query):
        if not self.api_key:
            return {"error": "API Key not found in .env file"}
        params = {'query': query, 'key': self.api_key, 'languageCode': 'en'}
        response = requests.get(self.base_url, params=params)
        return response.json() if response.status_code == 200 else {"error": "API call failed"}
