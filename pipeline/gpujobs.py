"""GPU job bundles for the stages that run on a borrowed GPU (Kaggle or Colab).

There is no GPU on the machine that runs `make film`, so the video and music stages work
in three steps:

  1. The stage finds the items whose cache key has no output yet and writes a bundle:
     outputs/jobs/<kind>/bundle/{jobs.json, inputs/...}. Job ids are "<item>-<hash>",
     the same names the stage looks for in its cache.
  2. kaggle/gpu_worker.py runs the bundle on the GPU. It writes results/<job id>.<ext>
     and results/runs.jsonl (one line per job: latency, GPU seconds, device, seed).
  3. `python -m pipeline.gpujobs import <kind> <results dir>` copies results whose name
     matches a pending job into outputs/<kind>/ and appends the worker's log lines to
     runs.jsonl. The next stage run then finds them in the cache.

Results that do not match a pending job are refused, so a stale download cannot slip an
old clip into the film.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .common import OUT, RUNS, PipelineError, say

KINDS = {"shots": "mp4", "music": "wav"}


def bundle_dir(kind: str) -> Path:
    return OUT / "jobs" / kind / "bundle"


def export(kind: str, jobs: list[dict], files: dict[str, Path], worker_cfg: dict) -> Path | None:
    """Write the bundle for pending jobs. Returns the bundle dir, or None if nothing is pending."""
    manifest = OUT / "jobs" / kind / "pending.json"
    if not jobs:
        if manifest.exists():
            manifest.unlink()
        return None
    b = bundle_dir(kind)
    if b.exists():
        shutil.rmtree(b)
    (b / "inputs").mkdir(parents=True)
    for name, src in files.items():
        shutil.copy2(src, b / "inputs" / name)
    doc = {"kind": kind, "ext": KINDS[kind], "worker": worker_cfg, "jobs": jobs}
    (b / "jobs.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8")
    # A committed copy of what was asked of the GPU, for the record.
    manifest.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return b


def import_results(kind: str, src: Path) -> int:
    manifest = OUT / "jobs" / kind / "pending.json"
    if not manifest.exists():
        raise PipelineError(f"no pending {kind} jobs; run the {kind} stage first")
    doc = json.loads(manifest.read_text("utf-8"))
    ext = doc["ext"]
    pending = {j["id"] for j in doc["jobs"]}
    dest = OUT / kind
    dest.mkdir(parents=True, exist_ok=True)

    got = 0
    for f in sorted(src.glob(f"*.{ext}")):
        if f.stem not in pending:
            say(f"  skip {f.name}: not a pending job (stale or renamed)")
            continue
        if f.stat().st_size == 0:
            say(f"  skip {f.name}: empty file")
            continue
        tmp = dest / (f.name + ".part")
        shutil.copy2(f, tmp)
        tmp.replace(dest / f.name)
        got += 1
        say(f"  imported {f.name}")

    log = src / "runs.jsonl"
    if log.exists():
        seen = set()
        if RUNS.exists():
            for line in RUNS.read_text("utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("worker_job_key"):
                    seen.add(r["worker_job_key"])
        added = 0
        with open(RUNS, "a", encoding="utf-8") as out:
            for line in log.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key = f"{r.get('item')}|{r.get('ts')}"
                if key in seen:
                    continue
                r["worker_job_key"] = key
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                added += 1
        say(f"  appended {added} worker log lines to runs.jsonl")
    missing = pending - {f.stem for f in dest.glob(f"*.{ext}")}
    say(f"{kind}: imported {got}; still missing {len(missing)}" + (f": {', '.join(sorted(missing))}" if missing else ""))
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import worker results")
    imp.add_argument("kind", choices=sorted(KINDS))
    imp.add_argument("dir", type=Path)
    a = ap.parse_args()
    try:
        import_results(a.kind, a.dir)
    except PipelineError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
