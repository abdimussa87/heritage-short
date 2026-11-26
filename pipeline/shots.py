"""Moving-shots stage: image-to-video with Wan 2.2 TI2V-5B on a Kaggle GPU.

A clip's cache key covers the keyframe file hash, the motion prompt, the seed, the model
and its params. Shots with no clip yet go into a GPU job bundle (see gpujobs.py). Until
their clips are imported, assembly uses a Ken Burns move on the keyframe instead, so the
film always builds.
"""
from __future__ import annotations

import argparse

from . import gpujobs
from .common import Stage, current_file, file_hash, load_env, load_storyboard, say

STAGE = "shots"
DEFAULTS = {"width": 1280, "height": 704, "num_frames": 121, "fps": 24, "steps": 50, "guidance": 5.0}


def params(sb: dict) -> dict:
    return {**DEFAULTS, **sb["models"]["video"].get("params", {})}


def run(sb: dict) -> Stage:
    st = Stage(STAGE)
    m, p = sb["models"]["video"], params(sb)
    jobs, files = [], {}
    for shot in sb["shots"]:
        sid, motion = shot["id"], shot.get("motion") or {}
        if not motion.get("enabled"):
            continue
        kf = current_file("keyframes", sid)
        if kf is None:
            say(f"  {sid}: no keyframe yet, skipped")
            continue
        seed = motion.get("seed", shot["seed"])
        inputs = {"model": m["id"], "params": p, "keyframe": file_hash(kf), "prompt": motion["prompt"], "seed": seed}
        path, h = st.existing(sid, inputs)
        if path:
            st.record(sid, path, h, inputs)
            say(f"  {sid}: cached {path.name}")
            continue
        job_id = f"{sid}-{h}"
        image_name = f"{job_id}{kf.suffix}"
        files[image_name] = kf
        jobs.append({"id": job_id, "item": sid, "image": image_name, "prompt": motion["prompt"], "seed": seed, "params": p})
        say(f"  {sid}: PENDING {job_id}")
    st.save_index()
    b = gpujobs.export(STAGE, jobs, files, {"model": m["id"], "license": m.get("license")})
    if b:
        say(f"shots: {len(jobs)} clips pending. Bundle: {b}\n"
            f"  Run kaggle/gpu_worker.py on it (see kaggle/README.md), then:\n"
            f"  uv run python -m pipeline.gpujobs import shots <results dir>\n"
            f"  Until then, assembly uses a Ken Burns move for these shots.")
    else:
        say(f"shots: all {len(st.index)} clips cached")
    return st


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    load_env()
    run(load_storyboard())


if __name__ == "__main__":
    main()
