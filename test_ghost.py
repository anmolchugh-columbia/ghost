#!/usr/bin/env python3
"""
Ghost closed-loop test harness.

Simulates the server.py pipeline exactly:
  background thread: chat_stream → _iter_sentences → sentence_q
  main loop:         sentence_q.get → synthesize_to_bytes → (would send to browser)

Metrics reported:
  TTFT              — time from chat_stream() start to first yielded token (LLM cold-start)
  time_to_first_audio — time from start to when first WAV is ready (what the user actually waits for)
  total             — wall time for full response

Run:
    cd ~/ghost && .venv/bin/python test_ghost.py

Reference targets (cloud production, e.g. Deepgram + GPT-4o + ElevenLabs):
    LLM TTFT          : 300-800ms
    Time to first audio: 750-1000ms
"""

import contextlib
import io
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

TEST_CASES = [
    # (query, label, should_search)
    ("What's the weather in Sunnyvale right now?",                                     "realtime / weather",        True),
    ("What do you know about the movie Obsession?",                                    "knowledge / movie",         True),
    ("Who is Andrej Karpathy?",                                                        "knowledge / person",        True),
    ("Hello, how are you doing today?",                                                "conversational / greeting", False),
    ("I need to wash my car. The car wash is 50 metres away. Should I walk or drive?", "reasoning / car wash",      False),
    ("What is 15 percent of 240?",                                                     "math / no search",          False),
]

# Mirrors server.py's _SENTENCE_RE / _iter_sentences exactly
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _iter_sentences(token_gen, min_len: int = 20):
    buf = ""
    for token in token_gen:
        buf += token
        parts = _SENTENCE_RE.split(buf)
        if len(parts) > 1:
            for part in parts[:-1]:
                part = part.strip()
                if len(part) >= min_len:
                    yield part
            buf = parts[-1]
    if buf.strip():
        yield buf.strip()


@dataclass
class Result:
    label: str
    query: str
    should_search: bool
    router_decision:      Optional[bool]  = None
    router_query:         Optional[str]   = None
    searched:             bool            = False
    ttft:                 Optional[float] = None   # first LLM token (router + search overhead included)
    time_to_first_audio:  Optional[float] = None   # first WAV ready (= TTFT + sentence gen + TTS synthesis)
    llm_total:            float           = 0.0    # full wall time
    response:             str             = ""
    first_sentence:       str             = ""

    @property
    def router_correct(self) -> bool:
        return self.router_decision == self.should_search


def run_test(query: str, label: str, should_search: bool) -> Result:
    """
    Mirrors the server.py pipeline:
      background thread → chat_stream → _iter_sentences → sentence_q
      main thread       → sentence_q.get → synthesize_to_bytes
    """
    from ghost.agent import chat_stream
    from ghost.tts import synthesize_to_bytes

    r = Result(label=label, query=query, should_search=should_search)
    history: list[dict] = []
    sentence_q: queue.Queue = queue.Queue()
    stdout_buf = io.StringIO()
    ttft_latch: list[Optional[float]] = [None]  # wall-clock time of first token

    def _capturing_stream():
        """Pass tokens through while latching the first arrival time."""
        with contextlib.redirect_stdout(stdout_buf):
            for token in chat_stream(history, query):
                if ttft_latch[0] is None:
                    ttft_latch[0] = time.perf_counter()
                yield token

    def generate():
        try:
            for sentence in _iter_sentences(_capturing_stream()):
                sentence_q.put(sentence)
        except Exception as exc:
            sentence_q.put(exc)
        finally:
            sentence_q.put(None)

    t0 = time.perf_counter()
    threading.Thread(target=generate, daemon=True).start()

    all_sentences: list[str] = []

    while True:
        item = sentence_q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            print(f"  [pipeline error] {item}", file=sys.stderr)
            break

        all_sentences.append(item)
        synthesize_to_bytes(item)

        if r.time_to_first_audio is None:
            r.time_to_first_audio = time.perf_counter() - t0

    r.llm_total    = time.perf_counter() - t0
    r.ttft         = (ttft_latch[0] - t0) if ttft_latch[0] is not None else None
    r.first_sentence = all_sentences[0] if all_sentences else ""
    r.response       = " ".join(all_sentences)

    # Parse captured stdout for router telemetry
    for line in stdout_buf.getvalue().splitlines():
        if line.startswith("[router]"):
            m = re.search(r"needs_search=(\w+)", line)
            if m:
                r.router_decision = m.group(1) == "True"
            m = re.search(r"query='([^']*)'", line)
            if m:
                r.router_query = m.group(1)
        elif line.startswith("[search]"):
            r.searched = True

    return r


def print_report(results: list[Result]) -> None:
    print(f"\n{'═'*80}")
    print("  GHOST PIPELINE LATENCY REPORT")
    print(f"{'═'*80}")

    for r in results:
        router_ok  = "✓" if r.router_correct else "✗"
        search_str = f"searched: {r.router_query!r:.50s}" if r.searched else "no search"
        ttft_s     = f"{r.ttft:.2f}s"                if r.ttft                is not None else "N/A"
        audio_s    = f"{r.time_to_first_audio:.2f}s"  if r.time_to_first_audio is not None else "N/A"
        total_s    = f"{r.llm_total:.2f}s"

        print(f"\n  [{r.label}]")
        print(f"  Q : {r.query}")
        print(f"  ┌─ TTFT {ttft_s}  │  1st audio {audio_s}  │  total {total_s}")
        print(f"  └─ Router {router_ok} {search_str}")
        preview = r.response[:150].replace("\n", " ")
        if len(r.response) > 150:
            preview += "…"
        print(f"  A : {preview}")

    # Aggregate stats
    ttfts   = [r.ttft               for r in results if r.ttft               is not None]
    audios  = [r.time_to_first_audio for r in results if r.time_to_first_audio is not None]
    totals  = [r.llm_total           for r in results]
    n_correct  = sum(1 for r in results if r.router_correct)
    n_searched = sum(1 for r in results if r.searched)

    print(f"\n{'─'*80}")
    print("  SUMMARY")
    if ttfts:
        print(f"  Avg TTFT            : {sum(ttfts)/len(ttfts):.2f}s   (target <0.8s  — first LLM token)")
    if audios:
        print(f"  Avg time-to-1st-audio: {sum(audios)/len(audios):.2f}s   (target <1.5s  — what user waits for)")
    print(f"  Avg total           : {sum(totals)/len(totals):.2f}s")
    print(f"  Searches fired      : {n_searched}/{len(results)}")
    print(f"  Router correct      : {n_correct}/{len(results)}")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    print("Ghost pipeline test — importing models (Kokoro warms up on first import)…")
    from ghost.tts import synthesize_to_bytes  # noqa: triggers Kokoro warmup
    print("Ready.\n")

    results: list[Result] = []
    for query, label, should_search in TEST_CASES:
        print(f"  [{label}] …", end=" ", flush=True)
        r = run_test(query, label, should_search)
        results.append(r)
        marker = "🔍" if r.searched else "  "
        ttft_s  = f"{r.ttft:.1f}s"               if r.ttft               is not None else "N/A"
        audio_s = f"{r.time_to_first_audio:.1f}s" if r.time_to_first_audio is not None else "N/A"
        print(f"{marker}  TTFT {ttft_s}  1st-audio {audio_s}  total {r.llm_total:.1f}s")

    print_report(results)
