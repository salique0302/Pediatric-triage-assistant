"""
app/llm.py

Groq LLM client for calling the triage model.
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT = 30.0  # seconds
TEMPERATURE = 0.1  # low for deterministic medical triage


class TriageLLMError(Exception):
    """Raised when LLM call fails"""
    pass


async def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Call Groq LLM API with system and user messages.

    Args:
        system_prompt: The system prompt (language-specific)
        user_message: The user message with symptoms and context

    Returns:
        Raw string content from the LLM response

    Raises:
        TriageLLMError: If API call fails or times out
    """
    if not GROQ_API_KEY:
        raise TriageLLMError("GROQ_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 1000,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(GROQ_URL, headers=headers, json=payload)

            if response.status_code != 200:
                error_text = response.text
                raise TriageLLMError(
                    f"Groq API error {response.status_code}: {error_text}"
                )

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content

    except httpx.TimeoutException:
        raise TriageLLMError(f"LLM call timed out after {TIMEOUT} seconds")
    except httpx.RequestError as e:
        raise TriageLLMError(f"LLM request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise TriageLLMError(f"Unexpected LLM response format: {str(e)}")
