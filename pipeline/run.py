"""Run the whole pipeline: validate -> narration -> keyframes -> shots -> music -> assemble.

Every stage reads storyboard.json and skips work whose cache key already has an output,
so a re-run after a small edit makes only the model calls that edit needs. With no API
keys, a clean clone still rebuilds the film from the committed outputs/.
"""
from __future__ import annotations

import argparse
import sys

from . import assemble, keyframes, music, narration, shots
from .common import PipelineError, load_env, load_storyboard, say

STAGES = ["narration", "keyframes", "shots", "music", "assemble"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radio", action="store_true", help="build the radio cut (narration + temp music on black)")
    ap.add_argument("--from", dest="start", choices=STAGES, default="narration")
    a = ap.parse_args()
    load_env()
    try:
        sb = load_storyboard()
        say("storyboard: valid")
        todo = ["narration", "assemble"] if a.radio else STAGES[STAGES.index(a.start):]
        for name in todo:
            say(f"\n== {name}")
            if name == "narration":
                narration.run(sb)
            elif name == "keyframes":
                keyframes.run(sb)
            elif name == "shots":
                shots.run(sb)
            elif name == "music":
                music.run(sb)
            elif name == "assemble":
                assemble.run_assembly(sb, radio=a.radio, aspects=("16x9",) if a.radio else ("16x9", "9x16"))
    except PipelineError as e:
        sys.exit(f"\nerror: {e}")


if __name__ == "__main__":
    main()
