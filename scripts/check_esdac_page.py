import requests
from bs4 import BeautifulSoup

url = "https://esdac.jrc.ec.europa.eu/content/soil-biodiversity-dna-bacteria-and-fungi"
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")
print("Links count:", len(soup.find_all("a")))
for a in soup.find_all("a", href=True):
    print(f"{a.text.strip()} -> {a['href']}")
