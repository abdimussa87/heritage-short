"""Smoke test: one small call to each hosted provider, logged to runs.jsonl.

Also lists the model ids each provider offers today, because names move fast and the
storyboard's `models` block must name ones that exist.
"""
from __future__ import annotations

import argparse
import os

from . import providers
from .common import OUT, PipelineError, call_model, load_env, say, write_atomic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-model", default=os.environ.get("SMOKE_TEXT_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--tts-model", default=os.environ.get("SMOKE_TTS_MODEL", "gemini-2.5-flash-preview-tts"))
    ap.add_argument("--image-model", default=os.environ.get("SMOKE_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell"))
    ap.add_argument("--list", action="store_true", help="only list available models")
    a = ap.parse_args()
    load_env()
    out = OUT / "smoke"
    ok = True

    if providers.provider_ready("gemini"):
        names = sorted(m["name"].removeprefix("models/") for m in providers.gemini_models())
        say("gemini models (tts / flash):")
        for n in names:
            if "tts" in n or "flash" in n:
                say(f"  {n}")
        if not a.list:
            try:
                txt = call_model(stage="smoke", provider="gemini", model=a.text_model, item="text", params={},
                                 fn=lambda: providers.gemini_text(a.text_model, "Reply with the single word: ready"))
                say(f"gemini text ok: {txt.strip()[:40]}")
                wav, rate = call_model(stage="smoke", provider="gemini", model=a.tts_model, item="tts", params={"voice": "Kore"},
                                       fn=lambda: providers.gemini_tts(a.tts_model, "Say warmly: Hello, this is a test.", "Kore"))
                write_atomic(out / "tts.wav", wav)
                say(f"gemini tts ok: {len(wav)} bytes at {rate} Hz -> outputs/smoke/tts.wav")
            except PipelineError as e:
                ok = False
                say(f"gemini FAILED: {e}")
    else:
        ok = False
        say("gemini: GEMINI_API_KEY not set, skipped")

    if providers.provider_ready("cloudflare"):
        say("cloudflare image models:")
        for m in providers.cloudflare_models("flux"):
            say(f"  {m.get('name')}")
        if not a.list:
            try:
                blob = call_model(stage="smoke", provider="cloudflare", model=a.image_model, item="image", seed=1, params={},
                                  fn=lambda: providers.cloudflare_image(a.image_model, "a clay pot on a wooden table, soft window light",
                                                                       seed=1, width=1024, height=576))
                ext = providers.image_ext(blob)
                write_atomic(out / f"image.{ext}", blob)
                say(f"cloudflare image ok: {len(blob)} bytes -> outputs/smoke/image.{ext}")
            except PipelineError as e:
                ok = False
                say(f"cloudflare FAILED: {e}")
    else:
        ok = False
        say("cloudflare: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set, skipped")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
