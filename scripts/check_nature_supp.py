import requests
import re

url = "https://www.nature.com/articles/s41467-023-37937-4"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
r = requests.get(url, headers=headers, timeout=30)
print("Status:", r.status_code)
links = re.findall(r"https://static-content\.springer\.com/esm/[^\s\"'<>]+", r.text)
for l in sorted(set(links)):
    print(l)

