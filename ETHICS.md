# Ethics: consent, licences, labelling

## Consent

Every real person shown or voiced needs a record here before anything of them is generated. `make validate` refuses a storyboard character with `real_person: true` unless this file has a heading with its `consent_id`.

<!-- Template (copy, fill in, keep the heading format exactly):

## CONSENT-<id>

- Person and relation:
- What they agreed to (likeness / voice / story / words used):
- How consent was given and recorded (date, form: voice note, message, signed note):
- They have seen the narration text: yes/no
- They may withdraw at any time; how they would tell me:
-->

## Private data

Free endpoints may use what is sent to them for training. Private family details (full names, addresses, health) stay out of prompts.

## Licences

| Component | Licence / terms | Notes |
|---|---|---|
| Wan 2.2 TI2V-5B | Apache 2.0 | |
| ACE-Step 1.5 | MIT; outputs usable commercially | |
| FLUX.1 schnell / FLUX.2 klein 4B | Apache 2.0 | FLUX.2 klein 9B is non-commercial; not used |
| Gemini text and TTS | Google API terms | check free-tier data use |
| FFmpeg | LGPL/GPL | tool, not distributed |

## Labelling

- The film ends on an end card that says it was made with AI (`storyboard.json` → `end_card`).
- The platform's AI disclosure is switched on when posting.
- Watermarks inside the signal (for example SynthID in Gemini audio) are not removed. Assembly re-encodes, which drops file-level metadata such as C2PA manifests from the final MP4; the source files in `outputs/` keep theirs.
