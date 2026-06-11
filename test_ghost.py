#!/usr/bin/env python3
"""
Ghost closed-loop test harness.

Calls chat_stream() directly — no browser, no audio — so latency at each
stage can be measured and iterated on.

Run:
    cd ~/ghost && .venv/bin/python test_ghost.py

Reference targets (cloud production, e.g. Deepgram + GPT-4o + ElevenLabs):
    LLM TTFT : 300-800ms
    TTS      : 75-200ms first audio chunk
    Total    : 750-1000ms end-to-end
"""

import contextlib
import io
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

TEST_CASES = [
    # (query, label, should_call_tool)
    ("What's the weather in Sunnyvale right now?",                                  "realtime / weather",      True),
    ("What do you know about the movie Obsession?",                                 "knowledge / movie",       True),
    ("Who is Andrej Karpathy?",                                                     "knowledge / person",      True),
    ("Hello, how are you doing today?",                                             "conversational / greeting", False),
    ("I need to wash my car. The car wash is 50 metres away. Should I walk or drive?", "reasoning / car wash", False),
    ("What is 15 percent of 240?",                                                  "math / no search",        False),
]


@dataclass
class Result:
    label: str
    query: str
    should_search: bool
    router_decision: Optional[bool]  = None   # what router said
    router_query:    Optional[str]   = None   # search query router produced
    searched:        bool            = False  # did a search actually run
    ttft:            Optional[float] = None   # time to first yielded token
    llm_total:       float           = 0.0   # total wall time for chat_stream
    tts_time:        Optional[float] = None   # time to synthesise first sentence
    response:        str             = ""
    first_sentence:  str             = ""

    @property
    def router_correct(self) -> bool:
        return self.router_decision == self.should_search


def run_test(query: str, label: str, should_search: bool) -> Result:
    from ghost.agent import chat_stream

    r = Result(label=label, query=query, should_search=should_search)
    history: list[dict] = []
    tokens: list[str] = []

    buf = io.StringIO()
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(buf):
        for token in chat_stream(history, query):
            if r.ttft is None:
                r.ttft = time.perf_counter() - t0
            tokens.append(token)

    r.llm_total = time.perf_counter() - t0
    r.response  = "".join(tokens).strip()

    # Parse captured stdout
    for line in buf.getvalue().splitlines():
        if line.startswith("[router]"):
            m_search = re.search(r"needs_search=(\w+)", line)
            m_query  = re.search(r"query='([^']*)'",    line)
            if m_search:
                r.router_decision = m_search.group(1) == "True"
            if m_query:
                r.router_query = m_query.group(1)
        elif line.startswith("[search]"):
            r.searched = True

    # First sentence for TTS timing
    sentences = re.split(r"(?<=[.!?])\s+", r.response)
    r.first_sentence = sentences[0] if sentences else r.response[:120]

    if r.first_sentence:
        from ghost.tts import synthesize_to_bytes
        t_tts = time.perf_counter()
        synthesize_to_bytes(r.first_sentence)
        r.tts_time = time.perf_counter() - t_tts

    return r


def print_report(results: list[Result]) -> None:
    print(f"\n{'═'*80}")
    print("  GHOST LATENCY REPORT")
    print(f"{'═'*80}")

    for r in results:
        router_ok = "✓" if r.router_correct else "✗"
        search_str = f"searched: {r.router_query!r:.55s}" if r.searched else "no search"
        ttft_s  = f"{r.ttft:.2f}s"     if r.ttft    is not None else "N/A"
        tts_s   = f"{r.tts_time:.2f}s" if r.tts_time is not None else "N/A"
        total_s = f"{r.llm_total:.2f}s"

        print(f"\n  [{r.label}]")
        print(f"  Q : {r.query}")
        print(f"  ┌─ TTFT {ttft_s}  │  total {total_s}  │  TTS(1st) {tts_s}")
        print(f"  └─ Router {router_ok} {search_str}")
        preview = r.response[:150].replace("\n", " ")
        if len(r.response) > 150:
            preview += "…"
        print(f"  A : {preview}")

    # Aggregate stats
    ttfts  = [r.ttft      for r in results if r.ttft    is not None]
    totals = [r.llm_total for r in results]
    ttss   = [r.tts_time  for r in results if r.tts_time is not None]
    n_correct  = sum(1 for r in results if r.router_correct)
    n_searched = sum(1 for r in results if r.searched)

    print(f"\n{'─'*80}")
    print("  SUMMARY")
    print(f"  Avg TTFT         : {sum(ttfts)/len(ttfts):.2f}s   (target <0.8s)")
    print(f"  Avg total        : {sum(totals)/len(totals):.2f}s")
    print(f"  Avg TTS/sentence : {sum(ttss)/len(ttss):.2f}s   (target <0.2s)")
    print(f"  Searches fired   : {n_searched}/{len(results)}")
    print(f"  Router correct   : {n_correct}/{len(results)}")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    print("Ghost test harness — importing models (Kokoro warms up on first import)…")
    from ghost.tts import synthesize_to_bytes  # noqa: triggers Kokoro warmup
    print("Ready.\n")

    results: list[Result] = []
    for query, label, should_search in TEST_CASES:
        print(f"  [{label}] …", end=" ", flush=True)
        r = run_test(query, label, should_search)
        results.append(r)
        marker = "🔍" if r.searched else "  "
        print(f"{marker}  TTFT {r.ttft:.1f}s  total {r.llm_total:.1f}s")

    print_report(results)
