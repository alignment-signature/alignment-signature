"""Pangram v3 API client, vendored from scripts/detection/evaluate_pangram.py.

Same interface as the upstream PangramClient — `check_text(text)` returns
a dict with `success`, `fraction_ai`, `prediction_short`, `error`, etc.
"""

from typing import Any

import requests

PANGRAM_API_URL = "https://text.api.pangram.com/v3"


class PangramClient:
    """Client for Pangram AI Detection API (v3)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between requests

    def check_text(self, text: str) -> dict[str, Any]:
        """Send text to Pangram v3 API and return detection results."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        payload = {"text": text}
        try:
            response = requests.post(PANGRAM_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "fraction_ai": data.get("fraction_ai"),
                    "fraction_ai_assisted": data.get("fraction_ai_assisted"),
                    "fraction_human": data.get("fraction_human"),
                    "prediction_short": data.get("prediction_short"),
                    "headline": data.get("headline"),
                    "raw_response": data,
                }
            elif response.status_code == 429:
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "retry_after": response.headers.get("Retry-After", 60),
                }
            elif response.status_code == 401:
                return {"success": False, "error": "Invalid API key"}
            else:
                return {"success": False, "error": f"API error {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
