"""
src/agent/agent.py
==================
LLM agent layer — Groq-powered intelligent recommendations.
Falls back to rule-based logic if the Groq API is unavailable.

Public API:
    generate_recommendation(station: str, demand: float) -> dict

Returns a structured dict:
    {
        "status":          str,        # "Normal" | "High Load" | "Overloaded"
        "recommendations": list[str],  # bullet points
        "reasoning":       str,        # 2-3 sentence explanation
        "raw":             str,        # raw model output (debug)
    }
"""


# This module combines machine learning demand predictions with LLM-based reasoning.
# It takes predicted EV charging demand as input and generates structured,
# actionable recommendations for infrastructure optimization.

import os
import re
from groq import Groq

# ------------------------------------------------------------------
# API key — reads GROQ_API_KEY env var, falls back to project key
# ------------------------------------------------------------------
_GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "YOUR_API_KEY",  # project default
)

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy singleton Groq client."""
    global _client
    if _client is None:
        if not _GROQ_API_KEY:
            raise EnvironmentError(
                "Groq API key not found. Set the GROQ_API_KEY environment variable."
            )
        _client = Groq(api_key=_GROQ_API_KEY)
    return _client


# ------------------------------------------------------------------
# System prompt — strictly controls output format
# ------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert EV infrastructure analyst specialising in charging network optimisation.

Given a station ID and its predicted next-day energy demand (kWh), respond with EXACTLY
this structured format — nothing before, nothing after:

STATUS: <Normal | High Load | Overloaded>

RECOMMENDATIONS:
• <specific, actionable recommendation 1>
• <specific, actionable recommendation 2>
• <specific, actionable recommendation 3>

REASONING:
<2-3 sentences explaining your load assessment, referencing the demand figure, and
describing what operational risks or opportunities exist>

Classification thresholds:
  Normal      →  demand < 50 kWh
  High Load   →  50 kWh ≤ demand < 150 kWh
  Overloaded  →  demand ≥ 150 kWh

Your recommendations must address:
  - Infrastructure adjustments (charger count, power levels)
  - Load balancing or time-of-use pricing strategies
  - Grid stress mitigation or renewable integration opportunities
"""


# ------------------------------------------------------------------
# Public function
# ------------------------------------------------------------------
def generate_recommendation(station: str, demand: float) -> dict:
    """
    Generate an intelligent, structured recommendation for the given EV station.

    Args:
        station: Station identifier (e.g. "CA-329")
        demand:  Predicted next-day energy demand in kWh

    Returns:
        dict with keys: status, recommendations (list), reasoning, raw
    """
    user_prompt = (
        f"Station {station} has a predicted next-day demand of {demand:.2f} kWh. "
        f"Provide your full structured assessment."
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=600,
        )
        raw_text = response.choices[0].message.content.strip()
        return _parse_response(raw_text)

    except EnvironmentError:
        raise  # Surface key-missing errors directly

    except Exception as exc:
        # API / network / quota error — fall back to rule-based logic
        return _rule_based_fallback(station, demand, error=str(exc))


# ------------------------------------------------------------------
# Rule-based fallback (used when Groq is unavailable)
# ------------------------------------------------------------------
def _rule_based_fallback(station: str, demand: float, error: str = "") -> dict:
    """
    Deterministic recommendation based on demand thresholds.
    Used automatically when the Groq API call fails.
    """
    if demand < 50:
        status = "Normal"
        recommendations = [
            "Maintain current charger configuration — demand is within normal range.",
            "Use off-peak hours to schedule preventive maintenance.",
            "Monitor for gradual demand growth over the next 30 days.",
        ]
        reasoning = (
            f"Station {station} is forecasted at {demand:.1f} kWh, well within the "
            f"normal operating threshold (<50 kWh). No immediate action is required."
        )
    elif demand < 150:
        status = "High Load"
        recommendations = [
            "Consider activating dynamic pricing to shift peak-hour demand.",
            "Pre-schedule at least one additional charger unit for the forecast day.",
            "Alert fleet operators to distribute charging across multiple time slots.",
        ]
        reasoning = (
            f"Station {station} is forecasted at {demand:.1f} kWh, indicating high "
            f"load (50–150 kWh range). Load-balancing measures should be activated "
            f"to prevent congestion and grid stress."
        )
    else:
        status = "Overloaded"
        recommendations = [
            "Immediately deploy additional fast-chargers or mobile charging units.",
            "Enforce strict time-slot booking to cap simultaneous sessions.",
            "Engage grid operator to secure extra capacity for the forecast day.",
        ]
        reasoning = (
            f"Station {station} is forecasted at {demand:.1f} kWh, which exceeds the "
            f"overload threshold (≥150 kWh). Infrastructure expansion and demand "
            f"curtailment are required to prevent service disruption."
        )

    note = f" (Rule-based fallback — Groq unavailable: {error})" if error else ""
    return {
        "status":          status,
        "recommendations": recommendations,
        "reasoning":       reasoning + note,
        "raw":             f"[fallback]{note}",
    }


# ------------------------------------------------------------------
# Internal parser
# ------------------------------------------------------------------
def _parse_response(text: str) -> dict:
    """Extract structured fields from the model's plain-text reply."""

    # STATUS
    status_m = re.search(r"STATUS:\s*(.+)", text, re.IGNORECASE)
    status   = status_m.group(1).strip() if status_m else "Unknown"

    # RECOMMENDATIONS (bullet lines following the header)
    recs_m = re.search(
        r"RECOMMENDATIONS:\s*\n((?:[ \t]*[•\-\*].+\n?)+)",
        text, re.IGNORECASE
    )
    if recs_m:
        recommendations = [
            line.lstrip("•-* \t").strip()
            for line in recs_m.group(1).splitlines()
            if line.strip()
        ]
    else:
        # Fallback: grab any bullet lines anywhere in the text
        bullets = re.findall(r"[•\-\*]\s+(.+)", text)
        recommendations = bullets if bullets else ["No specific recommendations generated."]

    # REASONING
    reasoning_m = re.search(r"REASONING:\s*([\s\S]+?)(?:\n\n|$)", text, re.IGNORECASE)
    reasoning   = reasoning_m.group(1).strip() if reasoning_m else "No reasoning provided."

    return {
        "status":          status,
        "recommendations": recommendations,
        "reasoning":       reasoning,
        "raw":             text,
    }
