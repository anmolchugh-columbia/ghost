import os

import requests

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


def web_search(query: str, max_results: int = 3) -> str:
    resp = requests.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json", "categories": "general"},
        timeout=8,
    ).json()
    results = resp.get("results", [])[:max_results]
    if not results:
        return "No results found."
    lines = []
    for r in results:
        snippet = (r.get("content") or "")[:250]
        lines.append(f"{r['title']}: {snippet}")
    return "\n\n".join(lines)
