"""Thin HTTP clients for the hosted providers. No SDKs, so a provider change is a small diff.

Each function does one request and raises RetryableError or PipelineError. Retries and
logging live in common.call_model, not here.
"""
from __future__ import annotations

import base64
import io
import os
import re
import wave

import requests

from .common import PipelineError, RetryableError, need_env, raise_for_http

GEMINI = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = (10, 180)


def _post(url: str, **kw) -> requests.Response:
    try:
        resp = requests.post(url, timeout=TIMEOUT, **kw)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise RetryableError(f"network: {e}") from e
    raise_for_http(resp)
    return resp


def _get(url: str, **kw) -> requests.Response:
    try:
        resp = requests.get(url, timeout=TIMEOUT, **kw)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise RetryableError(f"network: {e}") from e
    raise_for_http(resp)
    return resp


# ---------------------------------------------------------------- Gemini

def _gemini_headers() -> dict:
    (key,) = need_env("GEMINI_API_KEY")
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def gemini_models() -> list[dict]:
    out, token = [], None
    while True:
        params = {"pageSize": 1000, **({"pageToken": token} if token else {})}
        data = _get(f"{GEMINI}/models", headers=_gemini_headers(), params=params).json()
        out += data.get("models", [])
        token = data.get("nextPageToken")
        if not token:
            return out


def gemini_text(model: str, prompt: str, *, temperature: float = 0.4, json_mode: bool = False) -> str:
    cfg: dict = {"temperature": temperature}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}
    data = _post(f"{GEMINI}/models/{model}:generateContent", headers=_gemini_headers(), json=body).json()
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        raise PipelineError(f"unexpected Gemini response: {str(data)[:400]}") from e


def gemini_tts(model: str, text: str, voice: str) -> tuple[bytes, int]:
    """Returns (wav bytes, sample rate). Gemini returns raw 16-bit PCM; we wrap it in WAV."""
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    data = _post(f"{GEMINI}/models/{model}:generateContent", headers=_gemini_headers(), json=body).json()
    try:
        part = data["candidates"][0]["content"]["parts"][0]["inlineData"]
    except (KeyError, IndexError) as e:
        # A finishReason other than STOP (e.g. OTHER) sometimes means a transient TTS failure.
        reason = (data.get("candidates") or [{}])[0].get("finishReason")
        if reason and reason != "STOP":
            raise RetryableError(f"TTS returned no audio (finishReason={reason})") from e
        raise PipelineError(f"unexpected TTS response: {str(data)[:400]}") from e
    pcm = base64.b64decode(part["data"])
    m = re.search(r"rate=(\d+)", part.get("mimeType", ""))
    rate = int(m.group(1)) if m else 24000
    if len(pcm) < rate:  # under ~0.5 s of audio for a non-empty line
        raise RetryableError(f"TTS returned only {len(pcm)} bytes")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue(), rate


# ---------------------------------------------------------------- Cloudflare Workers AI

def _cf() -> tuple[str, dict]:
    acct, token = need_env("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
    return f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai", {"Authorization": f"Bearer {token}"}


def cloudflare_models(search: str = "flux") -> list[dict]:
    base, headers = _cf()
    return _get(f"{base}/models/search", headers=headers, params={"search": search}).json().get("result", [])


def cloudflare_image(model: str, prompt: str, *, seed: int, width: int, height: int,
                     steps: int | None = None, references: list[bytes] | None = None) -> bytes:
    """Text-to-image, optionally conditioned on reference images (FLUX.2 models only).

    FLUX.1 schnell takes JSON and returns base64 JPEG at a fixed size. FLUX.2 models take
    multipart form data, accept width/height and up to four input images.
    """
    base, headers = _cf()
    url = f"{base}/run/{model}"
    if "flux-2" in model:
        form = {"prompt": (None, prompt), "seed": (None, str(seed)), "width": (None, str(width)), "height": (None, str(height))}
        if steps:
            form["steps"] = (None, str(steps))
        for i, ref in enumerate((references or [])[:4]):
            form[f"input_image_{i}"] = (f"ref{i}.png", ref, "image/png")
        resp = _post(url, headers=headers, files=form)
    else:
        if references:
            raise PipelineError(f"{model} cannot take reference images; use a FLUX.2 model")
        body = {"prompt": prompt, "seed": seed, **({"steps": steps} if steps else {})}
        resp = _post(url, headers=headers, json=body)

    if resp.headers.get("content-type", "").startswith("image/"):
        return resp.content
    data = resp.json()
    img = (data.get("result") or {}).get("image")
    if not img:
        raise PipelineError(f"no image in Cloudflare response: {str(data)[:400]}")
    return base64.b64decode(img)


def image_ext(blob: bytes) -> str:
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if blob[:3] == b"\xff\xd8\xff":
        return "jpg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    raise PipelineError("unknown image format from provider")


def provider_ready(provider: str) -> bool:
    keys = {"gemini": ["GEMINI_API_KEY"], "cloudflare": ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]}[provider]
    return all(os.environ.get(k) for k in keys)
