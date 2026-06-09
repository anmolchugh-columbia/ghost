import os
import queue
import re
import struct
import subprocess
import tempfile
import threading
import wave

import numpy as np
from kokoro_onnx import Kokoro

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_BASE, "models", "kokoro-v1.0.onnx")
_VOICES = os.path.join(_BASE, "models", "voices-v1.0.bin")

VOICE = "af_heart"
SPEED = 1.0
LANG = "en-us"
MAX_CHARS = 150  # Kokoro can loop on long inputs; split at this threshold

_NORMALIZE = [
    (re.compile(r'(\d+)\s*°F\b'), r'\1 degrees Fahrenheit'),
    (re.compile(r'(\d+)\s*°C\b'), r'\1 degrees Celsius'),
    (re.compile(r'(\d+)\s*°'), r'\1 degrees'),
    (re.compile(r'(\d+)\s*%'), r'\1 percent'),
    (re.compile(r'\$(\d[\d,.]*)'), r'\1 dollars'),
    (re.compile(r'(\d+)\s*mph\b', re.IGNORECASE), r'\1 miles per hour'),
    (re.compile(r'(\d+)\s*km/h\b', re.IGNORECASE), r'\1 kilometers per hour'),
    (re.compile(r'\*{1,2}([^*]+)\*{1,2}'), r'\1'),
    (re.compile(r'`[^`]+`'), ''),
    (re.compile(r'#{1,6}\s+'), ''),
    (re.compile(r'\[([^\]]+)\]\([^\)]+\)'), r'\1'),
    (re.compile(r'[-*]\s+'), ''),
]

_CLAUSE_SPLIT = re.compile(r',\s+')


def clean_for_speech(text: str) -> str:
    for pattern, replacement in _NORMALIZE:
        text = pattern.sub(replacement, text)
    return text.strip()


def _chunk_text(text: str) -> list[str]:
    """Split long text at clause boundaries to stay under MAX_CHARS."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts = _CLAUSE_SPLIT.split(text)
    chunks, cur = [], ""
    for part in parts:
        candidate = cur + (", " if cur else "") + part
        if len(candidate) <= MAX_CHARS:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = part
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _fade(samples: np.ndarray, sr: int, ms: int = 8) -> np.ndarray:
    """Apply a short linear fade-in to remove leading transient clicks."""
    n = int(sr * ms / 1000)
    if len(samples) <= n:
        return samples
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = samples.copy()
    out[:n] *= ramp
    return out


def _write_wav(samples: np.ndarray, sr: int, path: str) -> None:
    samples = _fade(samples, sr)
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())


def _afplay(samples: np.ndarray, sr: int) -> None:
    """Play audio via macOS afplay — no PortAudio, no echo issues."""
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        _write_wav(samples, sr, path)
        subprocess.run(['afplay', path], check=True)
    finally:
        os.unlink(path)


class KokoroTTS:
    """
    Kokoro ONNX TTS with afplay-based playback (macOS native).

    say() synthesizes in the calling thread, hands audio to a background
    afplay process.  Sentence N plays while sentence N+1 is synthesized.
    finish() blocks until all queued audio has been spoken.
    """

    def __init__(self) -> None:
        self._kokoro = Kokoro(_MODEL, _VOICES)
        self._audio_q: queue.Queue = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._all_done = threading.Event()
        self._all_done.set()
        self._player = threading.Thread(target=self._play_loop, daemon=True)
        self._player.start()
        self._kokoro.create("Ready.", voice=VOICE, speed=SPEED, lang=LANG)

    def _play_loop(self) -> None:
        while True:
            samples, sr = self._audio_q.get()
            _afplay(samples, sr)
            with self._lock:
                self._pending -= 1
                if self._pending == 0:
                    self._all_done.set()

    def say(self, text: str) -> None:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        for chunk in _chunk_text(cleaned):
            with self._lock:
                self._pending += 1
                self._all_done.clear()
            samples, sr = self._kokoro.create(chunk, voice=VOICE, speed=SPEED, lang=LANG)
            self._audio_q.put((np.array(samples, dtype=np.float32), sr))

    def finish(self) -> None:
        self._all_done.wait()


def speak(text: str) -> None:
    tts = KokoroTTS()
    tts.say(text)
    tts.finish()
