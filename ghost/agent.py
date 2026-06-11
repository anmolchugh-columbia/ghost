from collections.abc import Generator
import json
import ollama
from ghost.context import SYSTEM_PROMPT
from tools.search import web_search

ROUTER_MODEL = "qwen3:0.6b"
MAIN_MODEL   = "qwen3:14b"

# Few-shot prompt — small models follow examples far more reliably than abstract rules.
_ROUTER_SYSTEM = """\
Decide if a web search is needed. Output JSON only, no other text.

Format: {"needs_search": true, "search_query": "..."} or {"needs_search": false, "search_query": ""}

Examples:
"What's the weather in NYC?" → {"needs_search": true, "search_query": "weather New York City today"}
"Who is Elon Musk?" → {"needs_search": true, "search_query": "Elon Musk"}
"What movies are out this week?" → {"needs_search": true, "search_query": "movies released this week"}
"Latest news on Apple?" → {"needs_search": true, "search_query": "Apple news today"}
"What is the capital of France?" → {"needs_search": true, "search_query": "capital of France"}
"Hello, how are you?" → {"needs_search": false, "search_query": ""}
"What is 15 percent of 240?" → {"needs_search": false, "search_query": ""}
"Should I walk or drive 50 metres?" → {"needs_search": false, "search_query": ""}
"What can you do?" → {"needs_search": false, "search_query": ""}
"Tell me a joke." → {"needs_search": false, "search_query": ""}\
"""


def route(query: str) -> tuple[bool, str]:
    """
    Ask the router model whether this query needs a web search.
    Returns (needs_search, search_query). Runs in ~200-400ms on qwen3:0.6b.
    Public so server.py can pre-route before spawning the generate thread.
    """
    resp = ollama.chat(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user",   "content": query},
        ],
        think=False,
        format={
            "type": "object",
            "properties": {
                "needs_search":  {"type": "boolean"},
                "search_query":  {"type": "string"},
            },
            "required": ["needs_search", "search_query"],
        },
    )
    try:
        data = json.loads(resp.message.content)
        return bool(data.get("needs_search")), data.get("search_query", "").strip()
    except Exception:
        return False, ""


def chat_stream(
    history: list[dict],
    user_input: str,
    routing: tuple[bool, str] | None = None,
) -> Generator[str, None, None]:
    """
    Route → (search) → stream.

    If `routing` is provided (pre-computed by the caller), the internal route()
    call is skipped — avoids double-routing when server.py pre-routes to send
    a filler phrase before the generate thread starts.
    """
    history.append({"role": "user", "content": user_input})

    # ── 1. Route ──────────────────────────────────────────────────────────────
    if routing is not None:
        needs_search, search_query = routing
    else:
        needs_search, search_query = route(user_input)
    print(f"[router] needs_search={needs_search}  query={search_query!r}", flush=True)

    # ── 2. Search ─────────────────────────────────────────────────────────────
    extra = ""
    if needs_search and search_query:
        try:
            results = web_search(search_query)
            print(f"[search] {search_query!r}", flush=True)
            extra = f"\n\nFresh web search results:\n{results}"
        except Exception as exc:
            print(f"[search error] {exc}", flush=True)

    # ── 3. Stream ─────────────────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT + extra}] + history
    stream = ollama.chat(
        model=MAIN_MODEL,
        messages=messages,
        think=False,
        stream=True,
    )

    accumulated = ""
    for chunk in stream:
        delta = chunk.message.content or ""
        if delta:
            accumulated += delta
            yield delta

    history.append({"role": "assistant", "content": accumulated})
