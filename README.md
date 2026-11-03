# agentic-composer

An AI agent that composes, synthesises, mixes and revises instrumental music as Python code,
steered by human listening. A LangGraph state machine runs the loop: a listening model hears
each render, Claude critiques and rewrites the piece, and the human's ratings and notes outrank
everything else.

```
listen -> human -> critique -> compose -> render -> listen -> ...
```

Every sound is made by code in this repository (a numpy synthesis engine), apart from one
sampled instrument, a CC-BY grand piano. Every version of every piece renders bit for bit from
the same file, so the whole revision history can be heard and compared.

## What it has made

Three original instrumentals, started from production principles associated with late-1970s and
early-1980s art-rock, electronic funk and ambient records (no existing melody, riff, recording or
arrangement is used):

| Piece | Brief | Now |
|---|---|---|
| A. Rivet Logic | angular electronic funk, 112 BPM | v3, rated "fire" by Jonathan |
| B. Lantern Arithmetic | generative miniature | v3 |
| C. Concrete Tide | bass-forward hybrid, 88 BPM | v6 onwards: a new piece in E that drifted towards early-90s house; piano-hook variations from v8 |

C shows the loop at work. Jonathan rated v4 5/10 ("very basic, no hook or groove") and a guest
rater gave v5 4/10 ("generic, start from scratch"). Measurement found the cause: the composer
was mixing blind, so its new guitar hook sat 14-16 dB under the bass. After it gained a
render-and-measure tool and a from-scratch mode, v6 scored 6.5 and v7 7 ("genuinely catchy"),
then v8 moved the hook to a sampled house piano. `DIARY.md` records every decision.

## How the agent works (`agent/`)

- **listen** (`listen.py`): Gemini hears the render, A/B against the previous revision, and
  returns scores and timestamped notes. Its scores move by a point or more between listens of
  the same file and it sometimes describes moments that do not exist, so it is treated as a
  noisy second opinion. The evaluation harness measures how well it agrees with human ratings.
- **human** (`graph.py`): in `human` mode the graph pauses (a LangGraph `interrupt`, checkpointed
  in SQLite) for your rating and notes, and resumes when you answer. In `auto` mode it passes
  through, and you rate afterwards with `agent.evaluate rate`.
- **critique** (`llm.py`): Claude reads the human notes and ratings first, then the listener's
  report, the measurements, the code and the rules, and returns a structured plan. With `--fresh`
  it designs a new composition instead of revising.
- **compose** (`llm.py`): Claude edits only the piece file, through a path-confined text editor.
  It can build new instruments and effects inside that file, and it has a `render_and_measure`
  tool that draft-renders the piece and reports every track's level against the mix (broadband,
  40-150 Hz and 300 Hz-4 kHz), so it balances parts by measurement. Every earlier revision is
  hash-checked: an edit that changes a version someone has heard is rejected.
- **render** (`tools.py`): renders and scores the new revision. True peak, clipping and length are
  hard gates; limiting and loudness are advisory, because the human ear has already overruled
  the limiter rule once.

Every run is traced in Langfuse (spans per node, generations with token usage, scores), and
`agent/evaluate.py` runs experiments asking whether the listening model agrees with the human
raters.

## Setup

Python 3.12 or later and ffmpeg on the PATH.

```
pip install -r requirements.txt
python samples/fetch_salamander.py      # the piano samples (about 280 MB kept), for C v8 onwards
cp .env.example .env                    # ANTHROPIC_API_KEY, GEMINI_API_KEY, Langfuse keys
python -m agent.check                   # confirms each service, without printing secrets
```

## Rendering

```
python render.py a 3                    # piece A at revision 3
python render.py all 3                  # every piece at revision 3
python render.py c 7 --stems            # also per-track stems
python render.py c 8 --preview          # draft render into out/<slug>/preview/
python render.py b 2 --analyse-only     # re-analyse an existing WAV
```

Each render writes the WAV (24-bit, 44.1 kHz; not kept in Git), the MIDI file, `score.json`
(every note with its synthesis parameters), `config.json` (patches, mix, buses, master),
`analysis.json` (loudness, frequency balance, rhythm, per-track levels) and `analysis.png`.

## Running the agent

```
python -m agent.run start c --iters 2 --mode human --budget 10
python -m agent.run resume <thread> --rating 7 --notes "the stop at bar 37 is great; dense 2 drags"
python -m agent.run start c --fresh --iters 1 --mode auto --budget 8 --notes "start over: ..."
python -m agent.evaluate rate c 7 7 --notes "superior; the hook needs more texture"
```

`--budget` caps the Claude spend in US dollars. `agent/ratings.json` holds the human ratings,
which the critic treats as ground truth.

## Layout

- `research/`: the toolchain audit and the production notes, which define the composition rules
  (R-numbers) and mix rules (M1-M3) the agent works to.
- `engine/`: synthesis (`dsp.py`, `instruments.py`, `fx.py`), the sampled piano (`sampler.py`),
  the song model, mixer, master chain and MIDI writer (`song.py`), the rules as functions
  (`rules.py`) and the analysis suite (`analysis.py`).
- `pieces/`: one file per piece. `build(rev)` returns the whole song for a revision; later
  revisions are gated on `rev`, so every version renders from the same file.
- `agent/`: the LangGraph agent and the Langfuse evaluation harness.
- `workflow/revise.js`: the earlier, non-listening revision loop (critic, producer, verifier) that
  made revision 3, kept for the record; it runs inside Claude Code's workflow tool.
- `out/<piece>/v<rev>/`: everything each render produces except the audio.
- `DIARY.md` (decisions and revisions), `COMPARISON.md` (first against final), `SAMPLES.md`
  (provenance and licences).

Rendering is deterministic: composition randomness and render noise are seeded from the piece
seed and revision 1, so a revision differs from the one before only by its stated changes. In the
author's workspace, renders run through a shared CPU budget (`scripts/cpu_budget.py` beside this
folder); without it they run directly.

## Licence

The code is MIT-licensed. The music (the compositions in `pieces/` and everything under `out/`)
is all rights reserved. The piano samples are CC-BY 3.0 (Salamander Grand Piano by Alexander
Holm) and are not distributed here. See `LICENSE` and `SAMPLES.md`.
