"""Shared pieces for every stage: storyboard loading, cache keys, the call log, retries.

Cache rule: every generated file is named <name>-<hash>.<ext>, where <hash> covers every
input that changes the output (prompt, seed, model, params, and the hashes of upstream
files). A stage regenerates a file only when no file with the current hash exists, so a
re-run after a one-line storyboard edit makes exactly one new model call. Each stage
writes outputs/<stage>/index.json, which tells the next stage which file is current.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNS = ROOT / "runs.jsonl"
SCHEMA = ROOT / "schema" / "storyboard.schema.json"
ETHICS = ROOT / "ETHICS.md"

RUN_ID = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


class PipelineError(Exception):
    """A failure the user must act on (missing key, bad storyboard, exhausted quota)."""


class RetryableError(Exception):
    """A failure worth another attempt (429, 5xx, timeout)."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env")


def need_env(*names: str) -> list[str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise PipelineError(f"missing {', '.join(missing)} (copy .env.example to .env and fill it in)")
    return [os.environ[n] for n in names]


# ---------------------------------------------------------------- storyboard

def storyboard_path() -> Path:
    return Path(os.environ.get("STORYBOARD") or ROOT / "storyboard.json")


def load_storyboard(path: Path | None = None) -> dict:
    path = path or storyboard_path()
    try:
        data = json.loads(path.read_text("utf-8"))
    except FileNotFoundError:
        raise PipelineError(f"{path} not found")
    except json.JSONDecodeError as e:
        raise PipelineError(f"{path} is not valid JSON: {e}")
    errors = validate_storyboard(data)
    if errors:
        raise PipelineError("storyboard is invalid:\n  - " + "\n  - ".join(errors))
    return data


def validate_storyboard(data: dict, ethics_text: str | None = None) -> list[str]:
    """Schema check plus the rules a schema cannot express. Returns a list of errors."""
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA.read_text("utf-8"))
    v = Draft202012Validator(schema)
    errors = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in v.iter_errors(data)]
    if errors:
        return errors

    shot_ids = [s["id"] for s in data["shots"]]
    char_ids = [c["id"] for c in data["characters"]]
    for dup in {i for i in shot_ids if shot_ids.count(i) > 1}:
        errors.append(f"shots: duplicate id {dup}")
    for dup in {i for i in char_ids if char_ids.count(i) > 1}:
        errors.append(f"characters: duplicate id {dup}")
    for s in data["shots"]:
        for c in s["characters"]:
            if c not in char_ids:
                errors.append(f"shots/{s['id']}: unknown character {c}")
        m = s.get("motion") or {}
        if m.get("enabled") and not m.get("prompt"):
            errors.append(f"shots/{s['id']}: motion.enabled needs motion.prompt")
    for cue in data["music"].get("cues", []):
        if cue["at_shot"] not in shot_ids:
            errors.append(f"music/cues: unknown shot {cue['at_shot']}")
    chosen = data["music"].get("chosen_seed")
    if chosen is not None and chosen not in data["music"]["seeds"]:
        errors.append("music/chosen_seed must be one of music/seeds")

    # Consent gate: a real person may not be generated without a record in ETHICS.md.
    if ethics_text is None:
        ethics_text = ETHICS.read_text("utf-8") if ETHICS.exists() else ""
    for c in data["characters"]:
        if c["real_person"]:
            cid = c.get("consent_id", "")
            if not re.search(rf"^#+\s*{re.escape(cid)}\s*$", ethics_text, re.M):
                errors.append(f"characters/{c['id']}: real person needs a '## {cid}' consent record in ETHICS.md")
    return errors


def mix(sb: dict) -> dict:
    """Mix settings with schema defaults filled in."""
    m = dict(sb.get("mix") or {})
    m.setdefault("lead_in_s", 1.0)
    m.setdefault("voice_head_s", 0.3)
    m.setdefault("voice_tail_s", 0.7)
    m.setdefault("music_gain_db", -8)
    duck = {"threshold": 0.02, "ratio": 6, "attack_ms": 300, "release_ms": 1500}
    duck.update(m.get("duck") or {})
    m["duck"] = duck
    m.setdefault("loudness_lufs", -14)
    m.setdefault("true_peak_db", -1)
    return m


# ---------------------------------------------------------------- hashing and cache

def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Stage:
    """Content-addressed outputs for one stage, plus its index.json."""

    def __init__(self, name: str):
        self.name = name
        self.dir = OUT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.index: dict[str, dict] = {}

    def path_for(self, item: str, inputs: dict, ext: str) -> tuple[Path, str]:
        h = stable_hash(inputs)
        return self.dir / f"{item}-{h}.{ext}", h

    def existing(self, item: str, inputs: dict) -> tuple[Path | None, str]:
        """Cached output for these inputs, whatever its extension, or None."""
        h = stable_hash(inputs)
        hits = sorted(p for p in self.dir.glob(f"{item}-{h}.*") if not p.name.endswith(".part"))
        return (hits[0] if hits else None), h

    def record(self, item: str, path: Path, h: str, inputs: dict, **extra: Any) -> None:
        self.index[item] = {"file": path.relative_to(ROOT).as_posix(), "hash": h, "inputs": inputs, **extra}

    def save_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=2, ensure_ascii=False) + "\n", "utf-8")
        tmp.replace(self.index_path)


def read_index(stage: str) -> dict[str, dict]:
    p = OUT / stage / "index.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


def current_file(stage: str, item: str) -> Path | None:
    entry = read_index(stage).get(item)
    if not entry:
        return None
    p = ROOT / entry["file"]
    return p if p.exists() else None


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file, so an interrupted run never leaves a half file that looks cached."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------- call log

def log_call(**fields: Any) -> None:
    """Append one line to runs.jsonl. One line per model call attempt, success or failure."""
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run_id": RUN_ID, **fields}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_model(
    *,
    stage: str,
    provider: str,
    model: str,
    params: dict,
    fn: Callable[[], Any],
    item: str | None = None,
    seed: int | None = None,
    attempts: int = 4,
    base_delay: float = 2.0,
    cost_usd: float = 0.0,
    list_price_usd: float | None = None,
) -> Any:
    """Run fn() with retries. Every attempt is logged with its latency and outcome."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        t0 = time.monotonic()
        try:
            result = fn()
        except RetryableError as e:
            last = e
            _log_attempt(stage, provider, model, params, item, seed, attempt, t0, "retry" if attempt < attempts else "error", str(e))
            if attempt < attempts:
                delay = e.retry_after if e.retry_after is not None else base_delay * 2 ** (attempt - 1)
                time.sleep(min(delay, 60) + random.uniform(0, 0.5))
            continue
        except Exception as e:
            _log_attempt(stage, provider, model, params, item, seed, attempt, t0, "error", f"{type(e).__name__}: {e}")
            raise
        _log_attempt(stage, provider, model, params, item, seed, attempt, t0, "ok", None, cost_usd, list_price_usd)
        return result
    raise PipelineError(f"{stage}/{item}: {model} failed after {attempts} attempts: {last}")


def _log_attempt(stage, provider, model, params, item, seed, attempt, t0, status, error, cost_usd=0.0, list_price_usd=None):
    log_call(
        stage=stage, item=item, provider=provider, model=model, params=params, seed=seed,
        attempt=attempt, status=status, latency_s=round(time.monotonic() - t0, 3),
        cost_usd=cost_usd if status == "ok" else 0.0, list_price_usd=list_price_usd if status == "ok" else None,
        error=error,
    )


def raise_for_http(resp) -> None:
    """Map an HTTP response to RetryableError / PipelineError."""
    if resp.status_code < 400:
        return
    body = resp.text[:500]
    if resp.status_code == 429 or resp.status_code >= 500:
        ra = resp.headers.get("Retry-After")
        raise RetryableError(f"HTTP {resp.status_code}: {body}", float(ra) if ra and ra.isdigit() else None)
    raise PipelineError(f"HTTP {resp.status_code}: {body}")


# ---------------------------------------------------------------- media helpers

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise PipelineError(f"command failed ({p.returncode}): {' '.join(cmd[:6])} ...\n{p.stderr[-2000:]}")
    return p


def media_duration(path: Path) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(p.stdout.strip())


def say(msg: str) -> None:
    print(msg, flush=True)
