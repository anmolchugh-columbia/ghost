from collections.abc import Generator
import ollama
from ghost.context import SYSTEM_PROMPT
from tools.search import web_search

MODEL = "qwen3:14b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    }
]


def run_tool(name: str, args: dict) -> str:
    if name == "web_search":
        print(f"[tool] web_search({args['query']!r})", flush=True)
        return web_search(args["query"])
    return f"Unknown tool: {name}"


def chat_stream(history: list[dict], user_input: str) -> Generator[str, None, None]:
    """
    Yields text tokens as the LLM generates them.

    First pass uses think='low' — a small reasoning budget that lets the model
    decide whether to call web_search and formulate a good query, without the
    15-20 second cost of full thinking. Follow-up passes (after tool results are
    in context) use think=False since synthesis needs no deliberation.

    First pass is buffered: we don't yield tokens until we know whether a tool
    call is coming, so hallucinated content never reaches TTS.
    """
    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # First pass — buffered, minimal thinking to drive tool-call decisions
    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        think="low",
        stream=True,
    )

    accumulated = ""
    tool_calls = []

    for chunk in stream:
        delta = chunk.message.content or ""
        if delta:
            accumulated += delta
        if chunk.message.tool_calls:
            tool_calls = chunk.message.tool_calls

    if not tool_calls:
        # No tool needed — release the buffer
        history.append({"role": "assistant", "content": accumulated})
        yield accumulated
        return

    # Tool call path — execute, then stream the final response
    while tool_calls:
        history.append({
            "role": "assistant",
            "content": accumulated,
            "tool_calls": [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            history.append({"role": "tool", "content": result})

        accumulated = ""
        tool_calls = []

        # Follow-up: stream directly, no thinking needed
        stream = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            think=False,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.message.content or ""
            if delta:
                accumulated += delta
                yield delta
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls

    history.append({"role": "assistant", "content": accumulated})
