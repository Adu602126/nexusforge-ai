"""
============================================================
NexusForge — Central Config & Gemini LLM Client
============================================================
Loads .env variables and provides a ready-to-use Gemini
client for all agents. Import this everywhere instead of
hardcoding keys.
============================================================
"""

import logging
import os

from dotenv import load_dotenv

# Load .env from project root (works from any subdirectory)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger("Config")

# ── Read keys ──────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "DEMO")
CAMPAIGNX_API_KEY     = os.getenv("CAMPAIGNX_API_KEY", "")
CAMPAIGNX_API_BASE_URL = os.getenv("CAMPAIGNX_API_BASE_URL", "")
ETHEREUM_PRIVATE_KEY  = os.getenv("ETHEREUM_PRIVATE_KEY", "")

# ── Gemini client factory ──────────────────────────────────
def get_gemini_client():
    """
    Returns a configured google.generativeai GenerativeModel.
    Falls back gracefully if the key is missing.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.warning("GEMINI_API_KEY not set — LLM features will use rule-based fallback.")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"Gemini client ready — model: {GEMINI_MODEL}")
        return model
    except ImportError:
        logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return None


def gemini_generate(prompt: str, max_tokens: int = 1024) -> str:
    """
    Simple one-shot text generation using Gemini.
    Returns the text response, or empty string on failure.
    """
    model = get_gemini_client()
    if not model:
        return ""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.7}
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return ""
