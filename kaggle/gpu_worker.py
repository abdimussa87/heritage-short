"""GPU worker for a job bundle made by pipeline/shots.py or pipeline/music.py.

Runs on Kaggle (2x T4) or Colab:

    python gpu_worker.py <bundle dir> <results dir> [--gpus 2]

* Writes results/<job id>.<ext> atomically and skips jobs whose result already exists,
  so a session that times out can be restarted and resumes where it stopped.
* With --gpus 2 it runs one process per GPU (the two T4s do not pool memory, so one big
  model per card is the useful way to use both).
* Appends one line per job to results/runs.jsonl: model, params, seed, latency, GPU
  seconds, device, dtype, status. `pipeline.gpujobs import` merges it into runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def log(results: Path, row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **row}
    with open(results / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def device_info():
    import torch
    name = torch.cuda.get_device_name(0)
    major, _ = torch.cuda.get_device_capability(0)
    # T4 (sm75) has no native bf16; emulated bf16 is very slow, so use fp16 there.
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return name, dtype


# ---------------------------------------------------------------- Wan 2.2 image-to-video

def load_wan(model_id: str, dtype):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()  # 16 GB T4: keep only the active part on the GPU
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    return pipe


def run_wan(pipe, job: dict, bundle: Path, out: Path) -> dict:
    import torch
    from diffusers.utils import export_to_video, load_image
    p = job["params"]
    image = load_image(str(bundle / "inputs" / job["image"])).convert("RGB").resize((p["width"], p["height"]))
    frames = pipe(
        image=image, prompt=job["prompt"], negative_prompt=p.get("negative_prompt", ""),
        height=p["height"], width=p["width"], num_frames=p["num_frames"],
        guidance_scale=p["guidance"], num_inference_steps=p["steps"],
        generator=torch.Generator("cpu").manual_seed(job["seed"]),
    ).frames[0]
    export_to_video(frames, str(out), fps=p["fps"])
    return {"frames": len(frames)}


# ---------------------------------------------------------------- ACE-Step 1.5 music
# The ACE-Step 1.5 Python API is not pinned here yet. Fill in run_acestep() on the first
# Kaggle run from https://github.com/ACE-Step/ACE-Step-1.5 and record what you found in
# FAILURES.md / EXPERIMENTS.md. The job fields are: caption, lyrics, bpm, key, duration_s,
# instrumental, seed.

def load_acestep(model_id: str, dtype):
    raise NotImplementedError("wire up ACE-Step 1.5 here (see comment above)")


def run_acestep(pipe, job: dict, bundle: Path, out: Path) -> dict:
    raise NotImplementedError


# ---------------------------------------------------------------- driver

def work(bundle: Path, results: Path, shard: int, nshards: int) -> int:
    doc = json.loads((bundle / "jobs.json").read_text("utf-8"))
    kind, ext, model_id = doc["kind"], doc["ext"], doc["worker"]["model"]
    jobs = [j for i, j in enumerate(doc["jobs"]) if i % nshards == shard]
    todo = [j for j in jobs if not (results / f"{j['id']}.{ext}").exists()]
    print(f"[gpu {shard}] {kind}: {len(todo)} of {len(jobs)} jobs to run", flush=True)
    if not todo:
        return 0
    dev, dtype = device_info()
    t0 = time.monotonic()
    pipe = (load_wan if kind == "shots" else load_acestep)(model_id, dtype)
    log(results, {"stage": kind, "item": "_load", "provider": "kaggle", "model": model_id, "status": "ok",
                  "latency_s": round(time.monotonic() - t0, 1), "gpu_s": round(time.monotonic() - t0, 1),
                  "device": dev, "dtype": str(dtype), "params": {}, "seed": None, "cost_usd": 0.0})
    failed = 0
    for j in todo:
        final = results / f"{j['id']}.{ext}"
        tmp = results / f"{j['id']}.part.{ext}"
        t0 = time.monotonic()
        row = {"stage": kind, "item": j["item"], "job": j["id"], "provider": "kaggle", "model": model_id,
               "params": j.get("params", {k: v for k, v in j.items() if k not in ("id", "item")}),
               "seed": j["seed"], "device": dev, "dtype": str(dtype), "cost_usd": 0.0}
        try:
            extra = (run_wan if kind == "shots" else run_acestep)(pipe, j, bundle, tmp)
            tmp.replace(final)
            row.update(status="ok", **extra)
        except Exception as e:  # keep going: one bad job must not cost the whole session
            failed += 1
            row.update(status="error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
        dt = round(time.monotonic() - t0, 1)
        row.update(latency_s=dt, gpu_s=dt)
        log(results, row)
        print(f"[gpu {shard}] {j['id']}: {row['status']} in {dt}s", flush=True)
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    if a.shard is not None or a.gpus == 1:
        sys.exit(1 if work(a.bundle, a.results, a.shard or 0, a.gpus) else 0)
    procs = []
    for g in range(a.gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        procs.append(subprocess.Popen([sys.executable, __file__, str(a.bundle), str(a.results),
                                       "--gpus", str(a.gpus), "--shard", str(g)], env=env))
    sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
