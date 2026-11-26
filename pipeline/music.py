"""Music stage: score candidates from ACE-Step 1.5 on a Kaggle GPU, one per seed.

The storyboard lists the seeds to audition and names the chosen one (music.chosen_seed).
Candidates with no file yet go into a GPU job bundle. Until a candidate is chosen and
present, assembly uses a quiet synthesized pad as a temp track.
"""
from __future__ import annotations

import argparse

from . import gpujobs
from .common import Stage, load_env, load_storyboard, say

STAGE = "music"


def inputs_for(sb: dict, seed: int) -> dict:
    mu, m = sb["music"], sb["models"]["music"]
    return {"model": m["id"], "params": m.get("params", {}), "caption": mu["caption"], "bpm": mu.get("bpm"),
            "key": mu.get("key"), "duration_s": mu["duration_s"], "instrumental": mu.get("instrumental", True),
            "lyrics": mu.get("lyrics", ""), "seed": seed}


def item_for(seed: int) -> str:
    return f"cand_{seed}"


def run(sb: dict) -> Stage:
    st = Stage(STAGE)
    jobs = []
    for seed in sb["music"]["seeds"]:
        inputs = inputs_for(sb, seed)
        item = item_for(seed)
        path, h = st.existing(item, inputs)
        if path:
            st.record(item, path, h, inputs)
            say(f"  {item}: cached {path.name}")
        else:
            jobs.append({"id": f"{item}-{h}", "item": item, **inputs})
            say(f"  {item}: PENDING")
    st.save_index()
    m = sb["models"]["music"]
    b = gpujobs.export(STAGE, jobs, {}, {"model": m["id"], "license": m.get("license")})
    chosen = sb["music"].get("chosen_seed")
    if b:
        say(f"music: {len(jobs)} candidates pending. Bundle: {b}\n"
            f"  Run kaggle/gpu_worker.py on it, then: uv run python -m pipeline.gpujobs import music <results dir>")
    if chosen is None:
        say("music: no chosen_seed yet, assembly uses the temp pad")
    elif item_for(chosen) not in st.index:
        say(f"music: chosen seed {chosen} has no file yet, assembly uses the temp pad")
    return st


def chosen_file(sb: dict):
    from .common import current_file
    seed = sb["music"].get("chosen_seed")
    return current_file(STAGE, item_for(seed)) if seed is not None else None


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    load_env()
    run(load_storyboard())


if __name__ == "__main__":
    main()
