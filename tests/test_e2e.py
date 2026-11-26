"""Builds a real film from synthetic media in a copy of the repo. Proves the FFmpeg graphs."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO / "pipeline", root / "pipeline")
    shutil.copytree(REPO / "schema", root / "schema")
    shutil.copy(REPO / "tests" / "fixtures" / "storyboard.json", root / "storyboard.json")
    out = root / "outputs"

    # fake narration: two tones, indexed the way the narration stage would index them
    (out / "narration").mkdir(parents=True)
    narr = {}
    for sid, secs, hz in [("s01", 4.0, 220), ("s02", 1.5, 330)]:
        f = out / "narration" / f"{sid}-test.wav"
        ff("-f", "lavfi", "-i", f"sine=f={hz}:d={secs}", "-ar", "24000", "-ac", "1", str(f))
        narr[sid] = {"file": f.relative_to(root).as_posix(), "hash": "test", "inputs": {}, "duration_s": secs}
    (out / "narration" / "index.json").write_text(json.dumps(narr))

    # fake keyframes for every shot and a fake 5 s clip for the motion shot
    (out / "keyframes").mkdir()
    kf = {}
    for sid in ["s01", "s02", "s03"]:
        f = out / "keyframes" / f"{sid}-test.png"
        ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:d=1", "-frames:v", "1", str(f))
        kf[sid] = {"file": f.relative_to(root).as_posix(), "hash": "test", "inputs": {}}
    (out / "keyframes" / "index.json").write_text(json.dumps(kf))
    (out / "shots").mkdir()
    clip = out / "shots" / "s02-test.mp4"
    ff("-f", "lavfi", "-i", "testsrc=s=1280x704:r=24:d=5", "-pix_fmt", "yuv420p", str(clip))
    (out / "shots" / "index.json").write_text(json.dumps({"s02": {"file": clip.relative_to(root).as_posix(), "hash": "t", "inputs": {}}}))

    subprocess.run([sys.executable, "-m", "pipeline.assemble"], cwd=root, check=True)
    return root


def probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_both_aspects_exported_with_matching_length(built):
    report = json.loads((built / "outputs/final/assembly.json").read_text())
    total = report["timeline"]["total"]
    for aspect, (w, h) in {"16x9": (1920, 1080), "9x16": (1080, 1920)}.items():
        f = built / f"outputs/final/film_{aspect}.mp4"
        s = probe(f, "stream=codec_name,width,height,r_frame_rate")["streams"]
        v = next(x for x in s if x["codec_name"] == "h264")
        assert (v["width"], v["height"]) == (w, h) and v["r_frame_rate"] == "24/1"
        assert any(x["codec_name"] == "aac" for x in s)
        dur = float(probe(f, "format=duration")["format"]["duration"])
        assert abs(dur - total) < 0.1


def test_sources_and_loudness(built):
    report = json.loads((built / "outputs/final/assembly.json").read_text())
    assert report["picture_sources"] == {"s01": "image", "s02": "clip", "s03": "image"}
    assert abs(report["audio"]["after_lufs"] - (-14)) < 1.0
    assert report["audio"]["after_true_peak_db"] <= -0.5
    assert (built / "outputs/final/subtitles_en.srt").read_text().count("-->") >= 2
