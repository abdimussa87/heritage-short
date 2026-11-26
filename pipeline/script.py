"""Script stage: checks the narration and fills missing English subtitles.

`check` (no model call) prints words and an estimated spoken length per shot, so the
narration can be cut to time before any audio exists.

`translate` asks the text model for an English subtitle for every shot whose
subtitle_en is empty, and writes it into storyboard.json. Machine translations are a
draft: a person who speaks the language reviews them before release. The call is logged
like every other model call.
"""
from __future__ import annotations

import argparse
import json

from . import providers
from .common import PipelineError, call_model, load_env, load_storyboard, say, storyboard_path

WORDS_PER_S = 2.2  # rough guide from the brief; measure the real rate from TTS output


def check(sb: dict) -> None:
    total_w = 0
    for s in sb["shots"]:
        w = len(s["narration"]["text"].split())
        total_w += w
        say(f"  {s['id']}: {w:3} words  ~{w / WORDS_PER_S:4.1f}s spoken  planned {s['duration_s']:4.1f}s")
    planned = sum(s["duration_s"] for s in sb["shots"]) + sb["end_card"]["duration_s"]
    say(f"script: {total_w} words, ~{total_w / WORDS_PER_S:.0f}s spoken, {planned:.0f}s planned incl. end card")


def translate(sb: dict, overwrite: bool = False) -> int:
    todo = [s for s in sb["shots"] if s["narration"]["text"].strip()
            and (overwrite or not (s["narration"].get("subtitle_en") or "").strip())]
    if not todo:
        say("translate: nothing to do")
        return 0
    m = sb["models"]["text"]
    lines = {s["id"]: s["narration"]["text"] for s in todo}
    prompt = (
        f"Translate each narration line from language '{sb['language']}' into natural spoken English for film "
        "subtitles. Keep the meaning and tone, keep names as they are, and keep each line short. "
        "Return only a JSON object mapping each id to its English line.\n\n" + json.dumps(lines, ensure_ascii=False)
    )
    raw = call_model(stage="script", provider="gemini", model=m["id"], item="translate",
                     params={"lines": len(lines), "language": sb["language"]},
                     fn=lambda: providers.gemini_text(m["id"], prompt, temperature=0.2, json_mode=True))
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PipelineError(f"translation was not JSON: {raw[:300]}") from e
    # Edit the file as text-level JSON, so the human-written order and content are kept.
    path = storyboard_path()
    doc = json.loads(path.read_text("utf-8"))
    n = 0
    for s in doc["shots"]:
        if s["id"] in out and isinstance(out[s["id"]], str):
            s["narration"]["subtitle_en"] = out[s["id"]].strip()
            n += 1
            say(f"  {s['id']}: {s['narration']['subtitle_en']}")
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8")
    say(f"translate: {n} subtitles written to {path.name}. Review them before release.")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "translate"])
    ap.add_argument("--overwrite", action="store_true", help="retranslate lines that already have a subtitle")
    a = ap.parse_args()
    load_env()
    sb = load_storyboard()
    check(sb) if a.cmd == "check" else translate(sb, a.overwrite)


if __name__ == "__main__":
    main()
