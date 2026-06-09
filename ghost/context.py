import os
from dotenv import load_dotenv

load_dotenv()

SECOND_BRAIN = os.environ.get("SECOND_BRAIN_PATH", "")
OWNER_NAME = os.environ.get("OWNER_NAME", "the user")
PROFILE_PATH = os.path.join(SECOND_BRAIN, "0 - Anmol/Meta/Profile.md")


def load_profile() -> str:
    try:
        with open(PROFILE_PATH) as f:
            return f.read()
    except FileNotFoundError:
        return ""


SYSTEM_PROMPT = f"""You are Ghost, a personal AI assistant for {OWNER_NAME}.
You run locally on their machine. You are direct, concise, and never add filler.
When you don't know something current or factual, use the web_search tool.

Here is {OWNER_NAME}'s profile for context:
{load_profile()}
"""
