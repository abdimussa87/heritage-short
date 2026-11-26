# Heritage Short

A pipeline that turns one file, `storyboard.json`, into a 60–90 second narrated short film with an original score, English subtitles and an AI end card, exported as 16:9 and 9:16.

> Story, audience and film links: see [DESIGN.md](DESIGN.md). (Filled in once the story is chosen.)

## Run it

You need [uv](https://docs.astral.sh/uv/) and FFmpeg (with libass, which most builds have).

```
git clone <this repo> && cd <repo>
make setup
make film
```

On a clean clone, `make film` needs no API keys. Every generated image, clip, voice line and score is committed in `outputs/`, and the pipeline finds each one by the hash of its inputs. It calls a model only for what you change. The result is in `outputs/final/film_16x9.mp4` and `outputs/final/film_9x16.mp4`.

To change the film, edit `storyboard.json` and run `make film` again. Only the shots whose inputs changed are regenerated, and each new model call adds one line to `runs.jsonl`. For that you need keys:

```
cp .env.example .env    # then fill in GEMINI_API_KEY, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
make smoke              # one call per provider, logged to runs.jsonl
```

Video clips and music run on a free Kaggle GPU. See [kaggle/README.md](kaggle/README.md). If a clip does not exist yet, the film uses a slow pan-and-zoom on the keyframe instead.

## Commands

| Command | What it does |
|---|---|
| `make film` | whole pipeline, both aspect ratios |
| `make radio-cut` | narration over a temp score on black, for user tests |
| `make validate` | check the storyboard against the schema and the consent rule |
| `make script` | words and estimated spoken length per shot |
| `make translate` | draft English subtitles for empty ones (review them by hand) |
| `make narration` / `keyframes` / `shots` / `music` / `assemble` | one stage |
| `make models` | list the model ids each provider offers today |
| `make test` | unit tests and an end-to-end build on synthetic media |

## What it costs

Nothing in money: every provider is on a free tier. The limits are daily image quota, Gemini rate limits and about 30 Kaggle GPU hours a week. Measured latency, GPU minutes and list-price equivalents are in [COSTS.md](COSTS.md).

## Where things are

| Path | Contents |
|---|---|
| `storyboard.json` | the contract: shots, durations, prompts, seeds, narration, music, mix |
| `schema/` | JSON Schema for the storyboard |
| `pipeline/` | one script per stage, plus `common.py` (cache, log, retries) |
| `kaggle/` | the GPU worker for clips and music |
| `runs.jsonl` | one line per model call attempt |
| `outputs/` | every generated file, kept for inspection |
| [ARCHITECTURE.md](ARCHITECTURE.md) | stages, contracts, caching, failure handling |
| [DESIGN.md](DESIGN.md) | audience, intent, user tests and what they changed |
| [EXPERIMENTS.md](EXPERIMENTS.md) | hypothesis, change, result |
| [FAILURES.md](FAILURES.md) | what broke and what fixed it |
| [COSTS.md](COSTS.md) | latency and GPU minutes per stage |
| [ETHICS.md](ETHICS.md) | consent, licences, labelling |
