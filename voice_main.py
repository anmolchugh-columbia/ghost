import re
from ghost.agent import chat_stream
from ghost.stt import record_with_vad, transcribe
from ghost.tts import KokoroTTS, speak

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def iter_sentences(token_gen, min_len: int = 20):
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


def run_voice_loop() -> None:
    print("Ghost is ready. Press Enter to speak, Ctrl+C to quit.\n")
    history = []
    tts = KokoroTTS()  # load model once; reuse across turns

    while True:
        try:
            input("[ Press Enter to speak ]")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        audio = record_with_vad()
        user_input = transcribe(audio)

        if not user_input:
            print("(nothing heard)\n")
            continue

        print(f"\nYou: {user_input}")

        if user_input.lower().strip(" .") in ("exit", "quit", "goodbye", "bye"):
            speak("Goodbye.")
            break

        print("Ghost: ", end="", flush=True)

        for sentence in iter_sentences(chat_stream(history, user_input)):
            print(sentence, end=" ", flush=True)
            tts.say(sentence)

        tts.finish()
        print("\n")


if __name__ == "__main__":
    run_voice_loop()
