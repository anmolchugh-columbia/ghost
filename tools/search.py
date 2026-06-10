from ddgs import DDGS

_MAX_BODY = 200  # chars per result — enough context, fewer LLM tokens


def web_search(query: str, max_results: int = 3) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    lines = []
    for r in results:
        body = r["body"][:_MAX_BODY]
        lines.append(f"{r['title']}: {body}")
    return "\n\n".join(lines)
