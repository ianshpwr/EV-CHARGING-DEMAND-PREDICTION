import os
import re
import logging
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_client: Groq | None = None

# ✅ UPDATED MODEL LIST (ONLY WORKING ONES)
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.3-8b-instant",
]

MAX_RETRIES = 2


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not _GROQ_API_KEY:
            raise EnvironmentError("Missing GROQ_API_KEY")
        _client = Groq(api_key=_GROQ_API_KEY)
    return _client


_SYSTEM_PROMPT = """\
You are an expert EV infrastructure analyst.

Return EXACT format:

STATUS: <Normal | High Load | Overloaded>

RECOMMENDATIONS:
• ...
• ...
• ...

REASONING:
<2-3 sentences>

Thresholds:
Normal < 50
High Load 50–150
Overloaded ≥150
"""


# ✅ CLEAN + RELIABLE LLM CALL
def _call_groq_with_resilience(client: Groq, user_prompt: str) -> str:
    last_error = None

    for model in CANDIDATE_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(f"Trying model: {model}")

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )

                logging.info(f"Success with model: {model}")
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                logging.warning(f"{model} failed (attempt {attempt+1}): {e}")
                time.sleep(1)

        logging.warning(f"Switching model from {model}")

    raise Exception(f"All models failed: {last_error}")


def generate_recommendation(station: str, demand: float) -> dict:
    user_prompt = (
        f"Station {station} has a predicted demand of {demand:.2f} kWh."
    )

    try:
        client = _get_client()
        raw_text = _call_groq_with_resilience(client, user_prompt)
        return _parse_response(raw_text)

    except Exception as exc:
        return _rule_based_fallback(station, demand, error=str(exc))


# ✅ FALLBACK (unchanged but cleaner message)
def _rule_based_fallback(station: str, demand: float, error: str = "") -> dict:
    if demand < 50:
        status = "Normal"
        recommendations = [
            "Maintain current setup",
            "Schedule maintenance off-peak",
            "Monitor usage trends",
        ]
        reasoning = f"Station {station} at {demand:.1f} kWh is within safe limits."

    elif demand < 150:
        status = "High Load"
        recommendations = [
            "Enable load balancing",
            "Add temporary chargers",
            "Shift demand via pricing",
        ]
        reasoning = f"Station {station} at {demand:.1f} kWh indicates high usage."

    else:
        status = "Overloaded"
        recommendations = [
            "Deploy more chargers",
            "Restrict usage slots",
            "Increase grid support",
        ]
        reasoning = f"Station {station} at {demand:.1f} kWh exceeds safe capacity."

    note = f" (Fallback used: {error})" if error else ""

    return {
        "status": status,
        "recommendations": recommendations,
        "reasoning": reasoning + note,
        "raw": "[fallback]",
    }


def _parse_response(text: str) -> dict:
    status_m = re.search(r"STATUS:\s*(.+)", text)
    status = status_m.group(1).strip() if status_m else "Unknown"

    bullets = re.findall(r"[•\-\*]\s+(.+)", text)
    recommendations = bullets if bullets else ["No recommendations"]

    reasoning_m = re.search(r"REASONING:\s*([\s\S]+)", text)
    reasoning = reasoning_m.group(1).strip() if reasoning_m else "No reasoning"

    return {
        "status": status,
        "recommendations": recommendations,
        "reasoning": reasoning,
        "raw": text,
    }