"""Validate storyboard.json against schema/storyboard.schema.json plus cross-field rules."""
from __future__ import annotations

import sys

from .common import PipelineError, load_storyboard, storyboard_path


def main() -> None:
    try:
        sb = load_storyboard()
    except PipelineError as e:
        sys.exit(f"error: {e}")
    planned = sum(s["duration_s"] for s in sb["shots"]) + sb["end_card"]["duration_s"]
    print(f"{storyboard_path().name}: valid, {len(sb['shots'])} shots, {planned:.0f}s planned")
    if not 60 <= planned <= 90:
        print(f"warning: planned length {planned:.0f}s is outside 60-90s (narration may still stretch it)")


if __name__ == "__main__":
    main()
