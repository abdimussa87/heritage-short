# One command: `make film`. Every stage reads storyboard.json and reuses cached outputs.
UV ?= uv run
PY := $(UV) python -m

.PHONY: help setup film radio-cut validate check smoke models script translate narration keyframes shots music assemble test lint clean-cache

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup: ## install Python deps (needs uv and ffmpeg)
	uv sync
	@command -v ffmpeg >/dev/null || (echo "ffmpeg not found: install it first" && exit 1)

film: ## run the whole pipeline and export both aspect ratios
	$(PY) pipeline.run

radio-cut: ## narration over temp music on black, for the first user test
	$(PY) pipeline.run --radio

validate: ## check storyboard.json against the schema
	$(PY) pipeline.validate

script: ## words and estimated spoken length per shot
	$(PY) pipeline.script check

translate: ## fill empty English subtitles with the text model
	$(PY) pipeline.script translate

smoke: ## call each provider once and log latency
	$(PY) pipeline.smoke

models: ## list model ids the providers offer today
	$(PY) pipeline.smoke --list

narration: ## TTS, one file per shot
	$(PY) pipeline.narration

keyframes: ## reference sheets, then one still per shot
	$(PY) pipeline.keyframes

shots: ## write the GPU bundle for clips not yet made
	$(PY) pipeline.shots

music: ## write the GPU bundle for score candidates not yet made
	$(PY) pipeline.music

assemble: ## edit, mix and export from what exists
	$(PY) pipeline.assemble

test: ## unit tests and an end-to-end assembly test on synthetic media
	$(UV) pytest -q

check: validate test ## everything CI would run

clean-cache: ## delete intermediate segments (never deletes generated media)
	rm -rf outputs/_cache
