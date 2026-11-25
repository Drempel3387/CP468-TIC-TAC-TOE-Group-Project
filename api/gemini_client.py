"""Small standard-library client for Google's Gemini generateContent API."""
import json
import os
import urllib.error
import urllib.request


class GeminiClient:
    def __init__(self, api_key=None, model=None, timeout=20, thinking_level=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.timeout = timeout
        self.thinking_level = thinking_level  # e.g. "minimal"/"low" for Gemini 3 models

    @property
    def is_configured(self):
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY.")
        endpoint = ("https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model}:generateContent?key={self.api_key}")
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.1,
                                        "responseMimeType": "application/json"}}
        if self.thinking_level:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail[:400]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not reach Gemini API: {exc}") from exc
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned an unexpected response format.") from exc
