"""Keyframes stage: character reference sheets first, then one still per shot.

Every shot prompt is built the same way: shot prompt + fixed character descriptions +
fixed style block. With a FLUX.2 model the character reference sheets go in as input
images too. A reference sheet's file hash is part of each dependent keyframe's cache key,
so a new reference sheet regenerates exactly the shots that show that character.
"""
from __future__ import annotations

import argparse

from . import providers
from .common import ROOT, PipelineError, Stage, call_model, file_hash, load_env, load_storyboard, say, write_atomic


def _params(sb: dict) -> dict:
    p = {"width": 1280, "height": 720, "ref_width": 1024, "ref_height": 1024, "steps": None}
    p.update(sb["models"]["image"].get("params", {}))
    return p


def _generate(stage: str, sb: dict, item: str, prompt: str, seed: int, w: int, h: int, refs: list[bytes]) -> bytes:
    m = sb["models"]["image"]
    if m["provider"] != "cloudflare":
        raise PipelineError(f"unsupported image provider {m['provider']}")
    p = _params(sb)
    return call_model(
        stage=stage, provider="cloudflare", model=m["id"], item=item, seed=seed,
        params={"width": w, "height": h, "steps": p["steps"], "n_refs": len(refs), "prompt_chars": len(prompt)},
        fn=lambda: providers.cloudflare_image(m["id"], prompt, seed=seed, width=w, height=h, steps=p["steps"], references=refs),
    )


def shot_prompt(sb: dict, shot: dict) -> str:
    chars = {c["id"]: c for c in sb["characters"]}
    parts = [shot["image_prompt"].strip()]
    parts += [chars[cid]["description"].strip() for cid in shot["characters"]]
    parts.append(sb["style"]["block"].strip())
    return " ".join(p if p.endswith(".") else p + "." for p in parts if p)


def run_references(sb: dict, retake: set[str] = frozenset()) -> Stage:
    st = Stage("references")
    m, p = sb["models"]["image"], _params(sb)
    for c in sb["characters"]:
        prompt = f"{c['reference_prompt'].strip()} {c['description'].strip()} {sb['style']['block'].strip()}"
        inputs = {"model": m["id"], "prompt": prompt, "seed": c["seed"], "w": p["ref_width"], "h": p["ref_height"], "steps": p["steps"]}
        path, h = st.existing(c["id"], inputs)
        status = "cached"
        if not path or c["id"] in retake:
            blob = _generate("references", sb, c["id"], prompt, c["seed"], p["ref_width"], p["ref_height"], [])
            path = st.dir / f"{c['id']}-{h}.{providers.image_ext(blob)}"
            write_atomic(path, blob)
            status = "NEW"
        st.record(c["id"], path, h, inputs)
        say(f"  ref {c['id']}: {status:6} {path.name}")
    st.save_index()
    return st


def run(sb: dict, retake: set[str] = frozenset()) -> Stage:
    refs = run_references(sb, retake)
    st = Stage("keyframes")
    m, p = sb["models"]["image"], _params(sb)
    use_refs = "flux-2" in m["id"]
    made = cached = 0
    for shot in sb["shots"]:
        sid = shot["id"]
        prompt = shot_prompt(sb, shot)
        ref_files = [ROOT / refs.index[c]["file"] for c in shot["characters"]] if use_refs else []
        inputs = {"model": m["id"], "prompt": prompt, "seed": shot["seed"], "w": p["width"], "h": p["height"],
                  "steps": p["steps"], "refs": [file_hash(f) for f in ref_files]}
        path, h = st.existing(sid, inputs)
        status = "cached"
        if path and sid not in retake:
            cached += 1
        else:
            blob = _generate("keyframes", sb, sid, prompt, shot["seed"], p["width"], p["height"], [f.read_bytes() for f in ref_files])
            path = st.dir / f"{sid}-{h}.{providers.image_ext(blob)}"
            write_atomic(path, blob)
            made += 1
            status = "NEW"
        st.record(sid, path, h, inputs)
        say(f"  {sid}: {status:6} {path.name}")
    st.save_index()
    say(f"keyframes: {made} generated, {cached} cached")
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retake", nargs="*", default=[], help="shot or character ids to regenerate even if cached")
    a = ap.parse_args()
    load_env()
    run(load_storyboard(), set(a.retake))


if __name__ == "__main__":
    main()
