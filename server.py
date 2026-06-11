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

_FILLER_WAV: bytes = b""  # pre-synthesized once at startup

STATIC = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_busy = False  # one conversation at a time


@app.on_event("startup")
async def _startup():
    global _FILLER_WAV
    loop = asyncio.get_running_loop()
    _FILLER_WAV = await loop.run_in_executor(None, synthesize_to_bytes, "Let me look that up.")

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

            # Pre-route so we can send filler immediately on search queries
            needs_search, search_query = await loop.run_in_executor(None, route, user_text)

            if needs_search:
                await ws.send_json({
                    "type": "sentence",
                    "text": "Let me look that up.",
                    "audio": base64.b64encode(_FILLER_WAV).decode(),
                })

            # LLM streams sentences into sentence_q (background thread)
            sentence_q: queue.Queue = queue.Queue()

            def generate():
                try:
                    for s in _iter_sentences(
                        chat_stream(history, user_text, routing=(needs_search, search_query))
                    ):
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
                    item = sentence_q.get()
                    if item is None or isinstance(item, Exception):
                        synthesis_q.put(item)
                        break
                    synthesis_q.put((item, synthesize_to_bytes(item)))

            threading.Thread(target=synthesize_loop, daemon=True).start()

            while True:
                item = await loop.run_in_executor(None, synthesis_q.get)
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
        _busy = False
