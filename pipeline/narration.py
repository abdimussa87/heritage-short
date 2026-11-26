"""Narration stage: one WAV per shot from Gemini TTS.

One file per shot keeps re-takes cheap: editing one line in storyboard.json changes one
hash, so one TTS call runs and every other line comes from the cache.
"""
from __future__ import annotations

import argparse
import wave

from . import providers
from .common import PipelineError, Stage, call_model, load_env, load_storyboard, say, write_atomic

STAGE = "narration"


def tts_text(sb: dict, shot: dict) -> str:
    style = sb["voice"]["style_prompt"].strip()
    text = shot["narration"]["text"].strip()
    return f"{style}\n\n{text}" if style else text


def inputs_for(sb: dict, shot: dict) -> dict:
    m = sb["models"]["tts"]
    return {"model": m["id"], "params": m.get("params", {}), "voice": sb["voice"]["name"],
            "language": sb["language"], "text": tts_text(sb, shot)}


def wav_seconds(path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def run(sb: dict, retake: set[str] = frozenset()) -> Stage:
    st = Stage(STAGE)
    m = sb["models"]["tts"]
    made = cached = 0
    for shot in sb["shots"]:
        sid = shot["id"]
        if not shot["narration"]["text"].strip():
            continue
        inputs = inputs_for(sb, shot)
        path, h = st.existing(sid, inputs)
        if path and sid not in retake:
            cached += 1
            status = "cached"
        else:
            if m["provider"] != "gemini":
                raise PipelineError(f"unsupported TTS provider {m['provider']}")
            wav, _rate = call_model(
                stage=STAGE, provider="gemini", model=m["id"], item=sid,
                params={"voice": inputs["voice"], "chars": len(inputs["text"])},
                fn=lambda: providers.gemini_tts(m["id"], inputs["text"], inputs["voice"]),
            )
            path = st.dir / f"{sid}-{h}.wav"
            write_atomic(path, wav)
            made += 1
            status = "NEW"
        dur = wav_seconds(path)
        st.record(sid, path, h, inputs, duration_s=round(dur, 3))
        say(f"  {sid}: {status:6} {dur:5.2f}s  {path.name}")
    st.save_index()
    total = sum(e["duration_s"] for e in st.index.values())
    say(f"narration: {made} generated, {cached} cached, {total:.1f}s of speech")
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retake", nargs="*", default=[], help="shot ids to regenerate even if cached")
    a = ap.parse_args()
    load_env()
    run(load_storyboard(), set(a.retake))


if __name__ == "__main__":
    main()
