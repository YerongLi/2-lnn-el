"""
    get_backlinks.py

    MediaWiki API Demos
    Demo of `Backlinks` module: Get request to list pages which link to a certain page.

    MIT License
"""

import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "list": "backlinks",
    "bltitle": "Germany",
	"bllimit": 50000,
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

BACKLINKS = DATA["query"]["backlinks"]

for b in BACKLINKS:
    print(b["title"])