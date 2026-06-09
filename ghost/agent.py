from collections.abc import Generator
import ollama
from ghost.context import SYSTEM_PROMPT
from tools.search import web_search

MODEL = "qwen3:8b"

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
        return web_search(args["query"])
    return f"Unknown tool: {name}"


def chat(history: list[dict], user_input: str) -> tuple[str, list[dict]]:
    history.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        tools=TOOLS,
        think=False,
    )

    msg = response.message

    # handle tool calls
    while msg.tool_calls:
        history.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        for tc in msg.tool_calls:
            result = run_tool(tc.function.name, tc.function.arguments)
            history.append({"role": "tool", "content": result})

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            think=False,
        )
        msg = response.message

    reply = msg.content or ""
    history.append({"role": "assistant", "content": reply})
    return reply, history


def chat_stream(history: list[dict], user_input: str) -> Generator[str, None, None]:
    """
    Yields text tokens as the LLM generates them.
    Handles tool calls internally — pauses stream, runs tool, resumes.
    History is updated in-place.
    """
    history.append({"role": "user", "content": user_input})

    while True:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        stream = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,
            stream=True,
        )

        accumulated = ""
        tool_calls = []

        for chunk in stream:
            delta = chunk.message.content or ""
            if delta:
                accumulated += delta
                yield delta
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls

        if not tool_calls:
            history.append({"role": "assistant", "content": accumulated})
            return

        # Tool call — execute and loop to stream the follow-up response
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
