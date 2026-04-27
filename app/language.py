"""
app/language.py

Language detection and prompt template loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from langdetect import detect, LangDetectException


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def detect_language(text: str) -> Literal["en", "ar"]:
    """
    Detect whether input text is English or Arabic.

    Defaults to "en" on detection failure to ensure safe fallback.
    """
    try:
        lang = detect(text)
        # langdetect returns "ar" for Arabic
        if lang == "ar":
            return "ar"
        return "en"
    except LangDetectException:
        return "en"


@lru_cache(maxsize=2)
def get_prompt_template(language: Literal["en", "ar"]) -> str:
    """
    Load and cache the system prompt template for the given language.

    Cached after first load — prompts don't change at runtime.
    """
    filename = "system_en.txt" if language == "en" else "system_ar.txt"
    filepath = PROMPTS_DIR / filename
    return filepath.read_text(encoding="utf-8")
