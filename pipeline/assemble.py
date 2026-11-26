"""Assembly stage: timeline, picture, mix, subtitles, end card, both aspect ratios. FFmpeg only.

Timeline: shots play back to back. Each shot lasts max(planned duration, head + narration
+ tail), so the voice sets the cut, as in a professional edit. The first shot's head is the
lead-in. The end card follows the last shot.

Picture per shot, in order of preference: the Wan clip, a Ken Burns move on the keyframe,
black (radio cut, or no keyframe yet). Vertical uses a crop of the 16:9 picture
(close shots) or the 16:9 picture over a blurred copy of itself (wide shots).

Sound: narration stem and music stem are each brought to a similar loudness, the music
is lowered by music_gain_db and ducked under the voice with sidechaincompress, the two
are summed, and a two-pass loudnorm sets the final level (-14 LUFS, -1 dBTP by default).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from .common import (OUT, ROOT, PipelineError, current_file, file_hash, load_env, load_storyboard, media_duration,
                     mix, read_index, run, say, stable_hash)
from .music import chosen_file

FPS = 24
CLIP_TRIM_S = 0.25          # Wan clips often have a weak first frames
MAX_SLOWDOWN = 1.25         # stretch a clip at most this much before holding its last frame
SIZES = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
CLOSE = {"close_up", "medium_close_up", "extreme_close_up", "insert"}
KB_CYCLE = ["in", "left", "out", "right"]
SEG_VERSION = 2             # bump when segment rendering changes, to invalidate the segment cache
SEG_DIR = OUT / "_cache" / "segments"


# ---------------------------------------------------------------- timeline

def _frames(seconds: float) -> float:
    """Round up to a whole number of frames, so picture and sound never drift apart."""
    import math
    return math.ceil(seconds * FPS - 1e-6) / FPS


def build_timeline(sb: dict, narr: dict[str, dict]) -> dict:
    m = mix(sb)
    t, shots = 0.0, []
    for i, shot in enumerate(sb["shots"]):
        sid = shot["id"]
        head = m["lead_in_s"] if i == 0 else m["voice_head_s"]
        voice = narr.get(sid, {}).get("duration_s", 0.0) if shot["narration"]["text"].strip() else 0.0
        need = head + voice + m["voice_tail_s"] if voice else 0.0
        dur = _frames(max(shot["duration_s"], need))
        shots.append({"id": sid, "start": t, "duration": dur, "voice_start": round(t + head, 3),
                      "voice_duration": round(voice, 3), "planned": shot["duration_s"]})
        t += dur
    card = {"start": t, "duration": _frames(sb["end_card"]["duration_s"])}
    total = t + card["duration"]
    return {"shots": shots, "end_card": card, "total": total}


# ---------------------------------------------------------------- subtitles

def wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def cues_for(sb: dict, tl: dict, width: int) -> list[tuple[float, float, list[str]]]:
    """Subtitle cues of at most two lines, timed in proportion to their share of characters."""
    by_id = {s["id"]: s for s in sb["shots"]}
    cues = []
    for row in tl["shots"]:
        text = (by_id[row["id"]]["narration"].get("subtitle_en") or "").strip()
        if not text or row["voice_duration"] <= 0:
            continue
        lines = wrap(text, width)
        groups = [lines[i:i + 2] for i in range(0, len(lines), 2)]
        total_chars = sum(len(" ".join(g)) for g in groups) or 1
        t = row["voice_start"]
        for g in groups:
            d = row["voice_duration"] * len(" ".join(g)) / total_chars
            cues.append((t, t + d, g))
            t += d
    return cues


def _ts_srt(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def _ts_ass(t: float) -> str:
    cs = round(t * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def _ass_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_srt(cues, path: Path) -> None:
    out = [f"{i}\n{_ts_srt(a)} --> {_ts_srt(b)}\n" + "\n".join(lines) + "\n" for i, (a, b, lines) in enumerate(cues, 1)]
    path.write_text("\n".join(out), "utf-8")


def write_ass(sb: dict, tl: dict, aspect: str, path: Path, subtitles: bool = True) -> None:
    W, H = SIZES[aspect]
    if aspect == "16x9":
        fs, ml, mr, mv, width = 52, 120, 120, 70, 42
    else:  # keep clear of the platform buttons on the right and the caption block at the bottom
        fs, ml, mr, mv, width = 50, 90, 150, 430, 30
    card_fs = round(fs * 1.1)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,DejaVu Sans,{fs},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,{ml},{mr},{mv},1
Style: Card,DejaVu Sans,{card_fs},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,{ml},{mr},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = []
    if subtitles:
        for a, b, lines in cues_for(sb, tl, width):
            text = "\\N".join(_ass_text(x) for x in lines)
            rows.append(f"Dialogue: 0,{_ts_ass(a)},{_ts_ass(b)},Sub,,0,0,0,,{text}")
    card = tl["end_card"]
    text = "\\N".join(_ass_text(x) for x in sb["end_card"]["lines"])
    rows.append(f"Dialogue: 1,{_ts_ass(card['start'] + 0.3)},{_ts_ass(tl['total'])},Card,,0,0,0,,{{\\fad(400,300)}}{text}")
    path.write_text(head + "\n".join(rows) + "\n", "utf-8")


# ---------------------------------------------------------------- picture

def _zoompan(kb: str, n: int, w: int, h: int) -> str:
    zmax = 1.12
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    z, x, y = "1", cx, cy
    if kb == "in":
        z = f"1+{zmax - 1:.3f}*on/{n}"
    elif kb == "out":
        z = f"{zmax}-{zmax - 1:.3f}*on/{n}"
    elif kb in ("left", "right", "up", "down"):
        z = str(zmax)
        frac = f"on/{n}" if kb in ("right", "down") else f"(1-on/{n})"
        if kb in ("left", "right"):
            x = f"(iw-iw/zoom)*{frac}"
        else:
            y = f"(ih-ih/zoom)*{frac}"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={FPS}"


def _vcrop(vx: float) -> str:
    return f"crop='min(iw,ih*9/16)':ih:'(iw-min(iw,ih*9/16))*{vx:.3f}':0"


BLUR_PAD = ("split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "boxblur=30:3,eq=brightness=-0.10[bg];[b]scale=1080:-2:flags=lanczos[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2")


def segment_graph(kind: str, aspect: str, fit: str, vx: float, kb: str, dur: float, src_dur: float | None) -> str:
    W, H = SIZES[aspect]
    n = round(dur * FPS)
    crop_v = aspect == "9x16" and fit == "crop"
    pad_v = aspect == "9x16" and fit == "blur_pad"
    if kind == "image":
        zw, zh = (W, H) if crop_v else (1920, 1080)
        up_w, up_h = int(zw * 1.5) // 2 * 2, int(zh * 1.5) // 2 * 2
        g = (_vcrop(vx) + "," if crop_v else "") + \
            f"scale={up_w}:{up_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={up_w}:{up_h}," + \
            _zoompan(kb, n, zw, zh)
    elif kind == "clip":
        avail = max((src_dur or 0) - CLIP_TRIM_S, 0.5)
        g = f"trim=start={CLIP_TRIM_S},setpts=PTS-STARTPTS"
        if dur > avail:
            g += f",setpts={min(dur / avail, MAX_SLOWDOWN):.4f}*PTS"
        g += f",fps={FPS},tpad=stop_mode=clone:stop_duration={dur:.3f},trim=duration={dur:.3f},setpts=PTS-STARTPTS"
        if crop_v:
            g += "," + _vcrop(vx) + f",scale={W}:{H}:flags=lanczos"
        else:
            g += ",scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080"
    else:
        raise ValueError(kind)
    if pad_v:
        g += "," + BLUR_PAD
    return f"[0:v]{g},setsar=1,format=yuv420p[out]"


def render_segment(source: tuple[str, Path | None], aspect: str, shot: dict, idx: int, dur: float) -> Path:
    kind, src = source
    W, H = SIZES[aspect]
    fit = shot.get("vertical_fit") or ("crop" if shot["shot_size"] in CLOSE else "blur_pad")
    vx = shot.get("vertical_x", 0.5)
    kb = shot.get("kenburns") or KB_CYCLE[idx % len(KB_CYCLE)]
    key = {"v": SEG_VERSION, "kind": kind, "src": file_hash(src) if src else None, "aspect": aspect,
           "fit": fit, "vx": vx, "kb": kb, "dur": dur}
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    out = SEG_DIR / f"{shot['id']}-{aspect}-{stable_hash(key)}.mp4"
    if out.exists():
        return out
    n = round(dur * FPS)
    enc = ["-frames:v", str(n), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", "-an"]
    tmp = out.with_name(out.stem + ".part.mp4")
    if kind == "black":
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={dur:.3f}", *enc, str(tmp)])
    else:
        src_dur = media_duration(src) if kind == "clip" else None
        graph = segment_graph(kind, aspect, fit, vx, kb, dur, src_dur)
        run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter_complex", graph, "-map", "[out]", *enc, str(tmp)])
    tmp.replace(out)
    return out


def picture_source(shot: dict, radio: bool) -> tuple[str, Path | None]:
    if radio:
        return ("black", None)
    if (shot.get("motion") or {}).get("enabled"):
        clip = current_file("shots", shot["id"])
        if clip:
            return ("clip", clip)
    kf = current_file("keyframes", shot["id"])
    return ("image", kf) if kf else ("black", None)


# ---------------------------------------------------------------- sound

def _loudnorm_measure(path: Path, target: float, tp: float) -> dict:
    p = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json",
             "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise PipelineError("could not parse loudnorm output")
    return json.loads(m.group(0))


def build_audio(sb: dict, tl: dict, narr: dict, work: Path) -> tuple[Path, dict]:
    m = mix(sb)
    total = tl["total"]
    # voice stem
    voice = work / "voice.wav"
    items = [(r, ROOT / narr[r["id"]]["file"]) for r in tl["shots"] if r["voice_duration"] > 0]
    if items:
        args, chains = [], []
        for i, (r, f) in enumerate(items):
            args += ["-i", str(f)]
            ms = round(r["voice_start"] * 1000)
            chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,adelay={ms}:all=1[a{i}]")
        graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(items))) + \
            f"amix=inputs={len(items)}:normalize=0,apad=whole_dur={total:.3f},loudnorm=I=-16:TP=-2:LRA=7,aresample=48000[v]"
        run(["ffmpeg", "-y", "-v", "error", *args, "-filter_complex", graph, "-map", "[v]", "-t", f"{total:.3f}",
             "-ac", "1", "-ar", "48000", str(voice)])
    else:
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{total:.3f}", str(voice)])

    # music stem: chosen candidate, or a quiet synthesized pad as the temp track
    music_src = chosen_file(sb)
    music = work / "music.wav"
    fade_out = max(total - 3.5, 0)
    shape = f"afade=t=in:d=2,afade=t=out:st={fade_out:.3f}:d=3.5,loudnorm=I=-16:TP=-2:LRA=11,volume={m['music_gain_db']}dB"
    if music_src:
        run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(music_src), "-t", f"{total:.3f}",
             "-af", f"aresample=48000,aformat=channel_layouts=stereo,{shape}", "-ar", "48000", str(music)])
        music_label = Path(music_src).name
    else:
        pad = ("aevalsrc='0.05*sin(2*PI*110*t)*(0.7+0.3*sin(2*PI*0.09*t))"
               "+0.04*sin(2*PI*164.81*t)*(0.6+0.4*sin(2*PI*0.06*t+1))"
               "+0.03*sin(2*PI*220*t)+0.02*sin(2*PI*277.18*t)*(0.5+0.5*sin(2*PI*0.05*t+2))'"
               f":s=48000:d={total:.3f}")
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", pad,
             "-af", f"lowpass=f=1500,pan=stereo|c0=c0|c1=c0,{shape}", "-ar", "48000", str(music)])
        music_label = "temp pad (synthesized)"

    # duck, sum, two-pass loudness
    d = m["duck"]
    mixed = work / "mix.wav"
    graph = (f"[0:a]aformat=channel_layouts=stereo,asplit=2[v1][vsc];"
             f"[1:a][vsc]sidechaincompress=threshold={d['threshold']}:ratio={d['ratio']}:"
             f"attack={d['attack_ms']}:release={d['release_ms']}[ducked];"
             f"[v1][ducked]amix=inputs=2:normalize=0[out]")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-i", str(music), "-filter_complex", graph,
         "-map", "[out]", "-ar", "48000", str(mixed)])
    I, TP = m["loudness_lufs"], m["true_peak_db"]
    meas = _loudnorm_measure(mixed, I, TP)
    final = work / "final.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(mixed), "-af",
         f"loudnorm=I={I}:TP={TP}:LRA=11:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:"
         f"measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}:"
         f"linear=true:print_format=summary,aresample=48000", "-ar", "48000", "-t", f"{total:.3f}", str(final)])
    check = _loudnorm_measure(final, I, TP)
    report = {"music": music_label, "target_lufs": I, "target_tp_db": TP,
              "before_lufs": float(meas["input_i"]), "after_lufs": float(check["input_i"]),
              "after_true_peak_db": float(check["input_tp"])}
    return final, report


# ---------------------------------------------------------------- main

def run_assembly(sb: dict, radio: bool = False, aspects=("16x9", "9x16"), subtitles: bool = True) -> dict:
    narr = read_index("narration")
    missing = [s["id"] for s in sb["shots"] if s["narration"]["text"].strip() and s["id"] not in narr]
    if missing:
        raise PipelineError(f"no narration for {', '.join(missing)}; run the narration stage first")
    tl = build_timeline(sb, narr)
    final_dir = OUT / ("radio_cut" if radio else "final")
    work = OUT / "_cache" / "work"
    for d in (final_dir, work):
        d.mkdir(parents=True, exist_ok=True)
    say(f"timeline: {len(tl['shots'])} shots + end card = {tl['total']:.1f}s")

    audio, loud = build_audio(sb, tl, narr, work)
    say(f"  audio: {loud['music']}, {loud['after_lufs']:.1f} LUFS, true peak {loud['after_true_peak_db']:.1f} dBTP")
    sources = {}
    exports = {}
    for aspect in aspects:
        W, H = SIZES[aspect]
        segs = []
        for i, (shot, row) in enumerate(zip(sb["shots"], tl["shots"])):
            src = picture_source(shot, radio)
            sources[shot["id"]] = src[0]
            segs.append(render_segment(src, aspect, shot, i, row["duration"]))
        segs.append(render_segment(("black", None), aspect, {"id": "endcard", "shot_size": "wide"}, 0, tl["end_card"]["duration"]))
        lst = work / f"concat_{aspect}.txt"
        lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), "utf-8")
        video = work / f"video_{aspect}.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])
        ass = work / f"subs_{aspect}.ass"
        write_ass(sb, tl, aspect, ass, subtitles)
        name = f"radio_cut_{aspect}.mp4" if radio else f"film_{aspect}.mp4"
        out = final_dir / name
        tmp = work / name
        # the ass filter path is relative to cwd, which avoids FFmpeg filter-escaping of absolute paths
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio),
             "-vf", f"ass={ass.name},fade=t=in:st=0:d=0.6", "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
             "-t", f"{tl['total']:.3f}", str(tmp)], cwd=work)
        shutil.move(tmp, out)
        exports[aspect] = {"file": out.relative_to(ROOT).as_posix(), "size": [W, H], "duration_s": round(media_duration(out), 2)}
        say(f"  wrote {out.relative_to(ROOT)} ({exports[aspect]['duration_s']}s)")

    write_srt(cues_for(sb, tl, 42), final_dir / "subtitles_en.srt")
    if radio:
        shutil.copy2(audio, final_dir / "radio_cut.wav")
    report = {"timeline": tl, "picture_sources": sources, "audio": loud, "exports": exports}
    (final_dir / "assembly.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    kinds = list(sources.values())
    say(f"assembly: {kinds.count('clip')} clips, {kinds.count('image')} Ken Burns, {kinds.count('black')} black")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radio", action="store_true", help="radio cut: narration + music on black, 16:9 only")
    ap.add_argument("--no-subtitles", action="store_true")
    a = ap.parse_args()
    load_env()
    sb = load_storyboard()
    run_assembly(sb, radio=a.radio, aspects=("16x9",) if a.radio else ("16x9", "9x16"), subtitles=not a.no_subtitles)


if __name__ == "__main__":
    main()
