"""
Ghost web server.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000
"""
import asyncio
import base64
import io
import queue
import re
import threading
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ghost.agent import chat_stream, route
from ghost.stt import transcribe
from ghost.tts import synthesize_to_bytes
from tools.search import web_search
from tools.searxng import start as _searxng_start, stop as _searxng_stop

_FILLER_WAV: bytes = b""  # pre-synthesized once at startup

STATIC = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_busy = False  # one conversation at a time


@app.on_event("startup")
async def _startup():
    global _FILLER_WAV
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _searxng_start)
    _FILLER_WAV = await loop.run_in_executor(None, synthesize_to_bytes, "Let me look that up.")


@app.on_event("shutdown")
async def _shutdown():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _searxng_stop)

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


def _decode_audio(raw: bytes) -> np.ndarray:
    """Decode WAV bytes sent from the browser (encoded client-side, no ffmpeg needed)."""
    buf = io.BytesIO(raw)
    with wave.open(buf) as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


@app.get("/")
async def index():
    return HTMLResponse((STATIC / "index.html").read_text())


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    global _busy
    await ws.accept()

    if _busy:
        await ws.send_json({"type": "error", "message": "Ghost is busy — try again in a moment."})
        await ws.close()
        return

    _busy = True
    history = []
    loop = asyncio.get_running_loop()
    stop = threading.Event()  # signals background threads to exit on disconnect

    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") != "audio":
                continue

            raw = base64.b64decode(data["data"])

            audio_np = await loop.run_in_executor(None, _decode_audio, raw)
            user_text = await loop.run_in_executor(None, transcribe, audio_np)

            if not user_text.strip():
                await ws.send_json({"type": "error", "message": "Couldn't hear anything — try again."})
                continue

            await ws.send_json({"type": "transcript", "text": user_text})

            # Pre-route so we can overlap search with filler when search is slow
            needs_search, search_query = await loop.run_in_executor(None, route, user_text)

            # Run search in background; only send filler if it takes > 1.5s
            prefetched: str | None = None
            if needs_search and search_query:
                search_task = loop.run_in_executor(None, web_search, search_query)
                try:
                    prefetched = await asyncio.wait_for(asyncio.shield(search_task), timeout=1.5)
                except asyncio.TimeoutError:
                    await ws.send_json({
                        "type": "sentence",
                        "text": "Let me look that up.",
                        "audio": base64.b64encode(_FILLER_WAV).decode(),
                    })
                    prefetched = await search_task

            # LLM streams sentences into sentence_q (background thread)
            sentence_q: queue.Queue = queue.Queue()

            def generate():
                try:
                    for s in _iter_sentences(
                        chat_stream(
                            history, user_text,
                            routing=(needs_search, search_query),
                            prefetched_results=prefetched,
                        )
                    ):
                        if stop.is_set():
                            break
                        sentence_q.put(s)
                except Exception as exc:
                    sentence_q.put(exc)
                finally:
                    sentence_q.put(None)

            threading.Thread(target=generate, daemon=True).start()

            # Synthesis thread: consumes sentence_q, synthesizes, feeds synthesis_q.
            # Sentence N+1 is synthesized while sentence N is being sent over the WebSocket.
            synthesis_q: queue.Queue = queue.Queue()

            def synthesize_loop():
                while True:
                    try:
                        item = sentence_q.get(timeout=0.5)
                    except queue.Empty:
                        if stop.is_set():
                            synthesis_q.put(None)
                            break
                        continue
                    if item is None or isinstance(item, Exception):
                        synthesis_q.put(item)
                        break
                    synthesis_q.put((item, synthesize_to_bytes(item)))

            threading.Thread(target=synthesize_loop, daemon=True).start()

            def _get_synthesized():
                while not stop.is_set():
                    try:
                        return synthesis_q.get(timeout=0.5)
                    except queue.Empty:
                        pass
                return None

            while True:
                item = await loop.run_in_executor(None, _get_synthesized)
                if item is None:
                    break
                if isinstance(item, Exception):
                    await ws.send_json({"type": "error", "message": str(item)})
                    break

                sentence, wav = item
                await ws.send_json({
                    "type": "sentence",
                    "text": sentence,
                    "audio": base64.b64encode(wav).decode(),
                })

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()  # unblock background threads immediately
        _busy = False
