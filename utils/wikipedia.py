import requests
import urllib.parse

def find_wikipedia_page_url(title):
    session = requests.Session()
    data = session.get(
        url="https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": title,
            "format": "json"
        }
    ).json()

    if len(data['query']['search']) > 0:
        urlencoded_title = urllib.parse.quote(data['query']['search'][0]['title'])
        return "http://en.wikipedia.org/wiki/" + urlencoded_title
    return None