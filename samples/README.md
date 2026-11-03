# Samples

Recorded samples used by `engine/sampler.py`. The audio files are not kept in Git; run
`python samples/fetch_salamander.py` to fetch them.

| Folder | Source | Author | Licence | Kept |
|---|---|---|---|---|
| `salamander/` | Salamander Grand Piano V3+20161209, 44.1 kHz 16-bit, from [FreePats](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) | Alexander Holm (Yamaha C5, two AKG C414s) | [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 120 notes: every minor third from A0 to C8, velocity layers 4, 8, 12 and 16 of 16 |

**Credit line for any release using the piano:** Piano samples: Salamander Grand Piano by
Alexander Holm, licensed under CC-BY 3.0.

The author's own README is kept as `salamander/AUTHOR_README.txt`. The samples are used
unaltered except for repitching (at most 1.5 semitones), trimming of leading silence, an applied
release envelope and the voice's EQ.
