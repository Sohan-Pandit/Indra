
import requests
import json

url = "http://localhost:8001/annotate"
headers = {
    "x-api-key": "PLACEHOLDER_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "abstract": "Climate change is increasing the frequency of extreme weather events.",
    "pub_year": "2023"
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
