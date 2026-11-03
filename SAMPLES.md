# Sample manifest

**One recorded instrument is used: a sampled grand piano**, from C v8 onwards. Every other
sound in every version is synthesised by code in this directory at render time.

| Sampled instrument | Where it is used | Source | Licence |
|---|---|---|---|
| Grand piano (`piano` voice, `engine/sampler.py`) | C v8 onwards | Salamander Grand Piano V3 by Alexander Holm (Yamaha C5), from FreePats; 120 of its notes kept (see `samples/README.md`) | CC-BY 3.0: credit "Piano samples: Salamander Grand Piano by Alexander Holm, CC-BY 3.0" |

The piano notes are repitched by at most 1.5 semitones, trimmed of leading silence, given a
release envelope and EQ; nothing else is altered. `samples/fetch_salamander.py` re-fetches them
from the original source.

Synthesised sounds:

| Sound family | Where it is made | Method | Provenance and licence |
|---|---|---|---|
| Synthesiser bass, motifs, leads, pads, organ | `engine/instruments.py` | band-limited (polyBLEP) oscillators, resonant filters, additive sines | original code in this project |
| Plucked guitar and bass | `engine/dsp.py` `ks_string` | Karplus-Strong string model excited by seeded noise | original code |
| Bells, glass, stabs | `engine/instruments.py` `fm_voice` | two- and three-operator phase modulation | original code |
| Struck metal (pipes, plates) | `engine/instruments.py` `metal` | modal synthesis: inharmonic partials with separate decays | original code; bar-mode ratios 1, 2.756, 5.404, 8.933, 13.34 are the textbook free-free beam modes |
| Drums (kick, snare, dropped-pitch snare, hats, clap, toms, rim, wood blocks, shaker, cowbell, noise hits) | `engine/instruments.py` `drum_*` | sine sweeps, filtered noise, square-wave clusters | original code; the hat's six square-wave frequencies follow the common analogue-cymbal recipe (frequencies, not audio) |
| Reverb impulse responses | `engine/fx.py` `make_ir` | seeded decaying filtered noise with early reflections | generated, not measured from any space |
| Tape hiss | `engine/fx.py` `tape` | seeded high-passed noise | generated |

Randomness (string excitations, noise, generative choices) comes from numpy's seeded generators,
so every render is reproducible from the code.

Third-party software used to make the audio: Python 3.13 (PSF licence), numpy 2.3 (BSD-3),
scipy 1.16 (BSD-3), matplotlib 3.10 (PSF-based, figures only), ffmpeg 8.1.2 (GPL build,
measurement only; no ffmpeg code or output is in the audio).

The MIDI files contain only notes written by this project's code.
