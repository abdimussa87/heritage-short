import copy
import json
from pathlib import Path

import pytest

from pipeline import assemble
from pipeline.common import stable_hash, validate_storyboard

FIX = json.loads((Path(__file__).parent / "fixtures" / "storyboard.json").read_text())


def test_fixture_is_valid():
    assert validate_storyboard(FIX, ethics_text="") == []


def test_hash_ignores_key_order_but_not_values():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_unknown_character_and_duplicate_shot_are_errors():
    sb = copy.deepcopy(FIX)
    sb["shots"][0]["characters"] = ["ghost"]
    sb["shots"][1]["id"] = "s01"
    errs = "\n".join(validate_storyboard(sb, ethics_text=""))
    assert "unknown character ghost" in errs and "duplicate id s01" in errs


def test_motion_without_prompt_is_an_error():
    sb = copy.deepcopy(FIX)
    del sb["shots"][1]["motion"]["prompt"]
    assert any("motion.prompt" in e for e in validate_storyboard(sb, ethics_text=""))


def test_real_person_needs_consent_record():
    sb = copy.deepcopy(FIX)
    sb["characters"][0]["real_person"] = True
    assert any("consent_id" in e for e in validate_storyboard(sb, ethics_text=""))  # schema: id required
    sb["characters"][0]["consent_id"] = "CONSENT-elder"
    assert any("consent record" in e for e in validate_storyboard(sb, ethics_text="no record"))
    assert validate_storyboard(sb, ethics_text="## CONSENT-elder\nAgreed on ...") == []
    # another person's record with a longer id must not count
    assert any("consent record" in e for e in validate_storyboard(sb, ethics_text="## CONSENT-elder-other\n"))


def test_chosen_music_seed_must_be_a_candidate():
    sb = copy.deepcopy(FIX)
    sb["music"]["chosen_seed"] = 99
    assert any("chosen_seed" in e for e in validate_storyboard(sb, ethics_text=""))


def test_timeline_stretches_for_voice_and_stays_on_frame_grid():
    narr = {"s01": {"duration_s": 4.0}, "s02": {"duration_s": 1.0}}
    tl = assemble.build_timeline(FIX, narr)
    s1, s2, s3 = tl["shots"]
    assert s1["duration"] >= 1.0 + 4.0 + 0.7          # lead-in + voice + tail beats the planned 3 s
    assert s2["duration"] == 3.0                         # planned length wins when the voice is short
    assert s3["voice_duration"] == 0
    for row in tl["shots"]:
        assert abs(row["duration"] * assemble.FPS - round(row["duration"] * assemble.FPS)) < 1e-6
    assert tl["total"] == pytest.approx(sum(r["duration"] for r in tl["shots"]) + tl["end_card"]["duration"])


def test_subtitle_cues_wrap_to_width_and_two_lines():
    tl = assemble.build_timeline(FIX, {"s01": {"duration_s": 4.0}, "s02": {"duration_s": 1.0}})
    cues = assemble.cues_for(FIX, tl, 20)
    assert all(len(line) <= 20 for _, _, lines in cues for line in lines)
    assert all(1 <= len(lines) <= 2 for _, _, lines in cues)
    s1 = tl["shots"][0]
    first = [c for c in cues if c[0] < s1["start"] + s1["duration"]]
    assert first[0][0] == pytest.approx(s1["voice_start"])
    assert first[-1][1] == pytest.approx(s1["voice_start"] + s1["voice_duration"])


def test_segment_graph_modes():
    g = assemble.segment_graph("image", "9x16", "crop", 0.3, "in", 3.0, None)
    assert "crop='min(iw,ih*9/16)'" in g and "s=1080x1920" in g
    g = assemble.segment_graph("clip", "9x16", "blur_pad", 0.5, "in", 6.0, 5.0)
    assert "setpts=1.2500*PTS" in g and "overlay" in g and "tpad" in g
