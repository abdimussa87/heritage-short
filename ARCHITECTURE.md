# Architecture

## One contract, six stages

`storyboard.json` is the only input. It holds the story, the look, the model ids, every prompt and seed, the narration lines and the mix settings. `schema/storyboard.schema.json` defines it, and `pipeline/validate.py` checks it before every run. Each stage is one script in `pipeline/`. It reads the storyboard and writes only to `outputs/<stage>/`.

```
storyboard.json ──validate──┬─> script     (check word counts; translate subtitles)      Gemini text
                            ├─> narration  (one WAV per shot)                              Gemini TTS
                            ├─> keyframes  (reference sheets, then one still per shot)     FLUX on Cloudflare
                            ├─> shots      (image-to-video clips)                          Wan 2.2 on Kaggle
                            ├─> music      (score candidates, one per seed)                ACE-Step on Kaggle
                            └─> assemble   (timeline, picture, mix, subtitles, exports)     FFmpeg, local
```

`make film` runs narration, keyframes, shots, music and assemble in that order (`pipeline/run.py`). `make radio-cut` runs narration and a black-screen assembly only.

## Rules the schema cannot express

`validate_storyboard()` in `pipeline/common.py` also checks these:

- Shot and character ids are unique, and every character a shot names exists.
- A shot with `motion.enabled` has a motion prompt.
- `music.chosen_seed` is one of the auditioned seeds.
- **Consent gate.** A character with `real_person: true` needs a `consent_id`, and `ETHICS.md` must contain a heading with that id. Without it, no stage runs.

## Caching

Each generated file is named `<item>-<hash>.<ext>`. The hash is a SHA-256 of every input that changes the output: model id, params, prompt, seed and, for downstream stages, the hash of the upstream file. Examples:

| Output | Cache key covers |
|---|---|
| narration `s03-….wav` | TTS model, voice, style prompt, language, line text |
| reference `elder-….png` | image model, reference prompt + description + style block, seed, size |
| keyframe `s03-….png` | image model, shot prompt + character descriptions + style block, seed, size, hashes of the reference sheets it uses |
| clip `s03-….mp4` | video model and params, keyframe file hash, motion prompt, seed |
| music `cand_11-….wav` | music model, caption, bpm, key, length, lyrics, seed |
| assembly segment | source file hash, aspect, fit, pan direction, duration, renderer version |

A stage calls a model only when no file with the current hash exists. A changed narration line therefore makes one TTS call. A new reference sheet regenerates exactly the keyframes that show that character, and then only their clips. Old files stay on disk, so changing a prompt back costs nothing.

Each stage writes `outputs/<stage>/index.json`. It maps each item to its current file, hash and inputs, and the next stage reads it. Files are written to `*.part` and renamed, so an interrupted run never leaves a half file that looks cached.

Because `outputs/` is committed, a clean clone rebuilds the film without any API key: every stage finds its files in the cache, and assembly runs locally.

## The GPU stages

The machine that runs `make film` has no GPU. Wan 2.2 and ACE-Step run on a Kaggle notebook (2× T4) in three steps:

1. `make shots` / `make music` writes a bundle of the pending jobs to `outputs/jobs/<kind>/bundle/` (`jobs.json` plus input images). Each job id is `<item>-<hash>`, the same name the stage looks for in its cache. A copy of the request is kept in `outputs/jobs/<kind>/pending.json`.
2. `kaggle/gpu_worker.py` runs the bundle. It runs one process per GPU, writes each result atomically and skips finished jobs, so a timed-out session resumes. It logs one line per job with latency, GPU seconds, device and dtype.
3. `uv run python -m pipeline.gpujobs import <kind> <results dir>` copies in results whose names match a pending job. It refuses others, so a stale download cannot slip an old clip into the film. It also merges the worker log into `runs.jsonl`.

Until a clip exists, assembly uses a Ken Burns move on the keyframe. Until a score is chosen, it uses a quiet synthesized pad. The film always builds.

## Assembly

- **Timeline.** Shots play back to back. A shot lasts `max(planned, head + narration + tail)`, rounded up to whole frames at 24 fps, so the voice sets the cut and picture never drifts from sound. `outputs/final/assembly.json` records the timeline.
- **Picture.** The source is the Wan clip if one exists, else a Ken Burns move on the keyframe, else black. For a clip, the first 0.25 s is trimmed, the clip is stretched at most 1.25×, and then its last frame is held. Vertical 9:16 uses a crop of the 16:9 picture for close shots (`vertical_x` sets the crop centre) and the 16:9 picture over a blurred copy of itself for wide shots. Each shot segment is cached.
- **Sound.** The voice stem and the music stem are each brought to about −16 LUFS. The music is lowered by `music_gain_db` and ducked under the voice with `sidechaincompress` (threshold 0.02, ratio 6, attack 300 ms, release 1500 ms by default). The stems are then summed, and a two-pass `loudnorm` sets −14 LUFS and −1 dBTP. The measured result is written to `assembly.json`.
- **Text.** English subtitles and the end card are rendered from one ASS file per aspect ratio: at most two lines per cue, about 42 characters per line for 16:9 and 30 for 9:16. The vertical subtitles sit above the bottom 22% of the frame and clear of the right-hand buttons. A sidecar `subtitles_en.srt` is exported too.
- **Export.** H.264 High, yuv420p, constant 24 fps, AAC 192 kbps at 48 kHz, faststart.

## Failure handling

- **Retries.** `common.call_model()` wraps every hosted call. HTTP 429, 5xx, timeouts and empty TTS audio get up to four attempts with exponential backoff and jitter, and the client honours `Retry-After`. Other 4xx errors fail at once with the provider's message.
- **Logging.** Every attempt, including failed ones, is one line in `runs.jsonl`: time, run id, stage, item, provider, model, params, seed, attempt, status, latency, cost and error.
- **Missing keys.** The error names the variable and points to `.env.example`. It appears only if a stage actually needs to call a model.
- **GPU sessions.** The worker skips finished jobs, catches per-job errors (one bad clip does not end the session) and frees GPU memory after an out-of-memory error.
- **Fallbacks.** Ken Burns for a missing clip, a temp pad for missing music, black for a missing keyframe in the radio cut.

## Secrets

Keys live in `.env`, which git ignores. `.env.example` lists the variables. Nothing in `outputs/` or `runs.jsonl` contains a key: the clients send keys in headers, not URLs, and log only model ids and params.
