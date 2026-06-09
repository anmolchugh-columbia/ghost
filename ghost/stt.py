import collections
import numpy as np
import sounddevice as sd
import webrtcvad
import mlx_whisper

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480 samples
SILENCE_TIMEOUT_FRAMES = 25    # ~750ms of silence triggers stop
VAD_AGGRESSIVENESS = 1         # 0-3; 1 is permissive, good for normal speech
WHISPER_MODEL = "mlx-community/whisper-small-mlx"


def record_with_vad() -> np.ndarray:
    """Record audio, stopping automatically after ~1.2s of silence."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames: list[np.ndarray] = []
    silence_count = 0
    speech_started = False
    ring_buffer: collections.deque = collections.deque(maxlen=10)

    def callback(indata, frame_count, time, status):
        nonlocal silence_count, speech_started
        frame = indata[:, 0].copy()                         # float32
        pcm = (frame * 32767).astype(np.int16).tobytes()   # int16 PCM for VAD
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)

        if not speech_started:
            ring_buffer.append((frame, is_speech))
            if sum(1 for _, s in ring_buffer if s) > 0.5 * ring_buffer.maxlen:
                speech_started = True
                for buffered_frame, _ in ring_buffer:
                    frames.append(buffered_frame)
                ring_buffer.clear()
        else:
            frames.append(frame)
            if is_speech:
                silence_count = 0
            else:
                silence_count += 1

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",        # float32 input — convert to int16 only for VAD
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        print("(listening...)", end=" ", flush=True)
        while True:
            sd.sleep(FRAME_MS)
            if speech_started and silence_count >= SILENCE_TIMEOUT_FRAMES:
                break

    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames)


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, verbose=False)
    return result["text"].strip()
