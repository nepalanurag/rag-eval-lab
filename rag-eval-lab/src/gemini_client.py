"""Minimal cached Gemini text-generation client.

Auth: dynamic credential surrogate via add_surrogate_to_request, restricted to
generativelanguage.googleapis.com. Never prints, logs, or persists credentials;
the cache stores only model responses keyed by a hash of (model, prompt).
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import (  # noqa: E402
    add_surrogate_to_request,
    read_json_response,
    DynamicCredentialError,
)

CREDENTIAL = "custom.google-gemini"
HOSTS = ["generativelanguage.googleapis.com"]
BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-2.5-flash"  # flash tier, text generation

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "cache",
)


def _cache_key(model, prompt, temperature):
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update(str(temperature).encode())
    return h.hexdigest()[:32]


def _cache_path(key):
    return os.path.join(CACHE_DIR, key + ".json")


def generate(prompt, temperature=0.2, max_output_tokens=1024, retries=8,
             pace_seconds=7.0):
    """Return generated text. Cached on disk; sequential use expected.

    pace_seconds: minimum seconds between API calls (stays under RPM quota).
    429 responses trigger a 65s cooldown before retrying.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(MODEL, prompt, temperature)
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["text"]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    url = f"{BASE}/models/{MODEL}:generateContent"
    last_err = None
    for attempt in range(retries):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            add_surrogate_to_request(req, CREDENTIAL, allowed_hosts=HOSTS)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = read_json_response(resp)
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            with open(path, "w") as f:
                json.dump({"model": MODEL, "prompt_hash": key, "text": text}, f)
            time.sleep(pace_seconds)  # stay under the RPM quota
            return text
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {exc.code}: {detail[:300]}"
            if exc.code == 429:
                delay = _retry_delay_seconds(detail)
                if delay is not None and delay > 300:
                    raise QuotaExhausted(
                        f"daily quota exhausted for {MODEL}: retry in {delay}s"
                    )
                time.sleep(65)  # short RPM cooldown, then retry
                continue
        except (DynamicCredentialError, KeyError, IndexError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_err}")


class QuotaExhausted(RuntimeError):
    """Raised when the API reports a daily/quota exhaustion (long retryDelay)."""


def set_model(name):
    global MODEL
    MODEL = name


def _retry_delay_seconds(detail):
    """Extract retryDelay (e.g. '64944s') from a 429 error body, else None."""
    import re

    m = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', detail)
    return int(m.group(1)) if m else None


def regenerate(prompt, **kwargs):
    """Delete the cached entry (if any) and fetch a fresh response."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(MODEL, prompt, kwargs.get("temperature", 0.2))
    path = _cache_path(key)
    if os.path.exists(path):
        os.remove(path)
    return generate(prompt, **kwargs)


if __name__ == "__main__":
    print(generate("Reply with exactly: OK"))
