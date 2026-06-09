import json
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
