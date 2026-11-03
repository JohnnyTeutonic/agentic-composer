# First against final (v1 against v3)

v3 is the current final for each piece. Jonathan heard A v3 and rated it highly; B v3 and C v3
await his listening. All values come from each render's `analysis.json`; loudness, LRA and true
peak are ffmpeg's EBU R128 measurements.

| Piece | Version | Length (s) | Loudness (LUFS) | LRA (LU) | True peak (dBTP) | Limiting (dB) | Centroid (Hz) | Mid + presence (%) | Stereo corr. | Corr. below 150 Hz | Adjacent-bar similarity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Rivet Logic | v1 | 147 | −14.5 | 2.6 | −1.1 | 4.0 | 325 | 5.1 | 0.98 | 1.00 | 0.53 |
| A. Rivet Logic | v3 | 147 | −14.6 | 2.9 | −1.9 | 6.5 | 688 | 15.1 | 0.94 | 1.00 | 0.45 |
| B. Lantern Arithmetic | v1 | 155 | −16.9 | 3.8 | −4.1 | 0 | 149 | 0.6 | 0.21 | 0.18 | 0.45 |
| B. Lantern Arithmetic | v3 | 155 | −16.9 | 8.3 | −2.9 | 0 | 371 | 6.0 | 0.58 | 0.96 | 0.46 |
| C. Concrete Tide | v1 | 141 | −13.9 | 6.4 | −1.4 | 3.5 | 638 | 20.9 | 0.67 | 0.82 | 0.61 |
| C. Concrete Tide | v3 | 141 | −13.9 | 6.6 | −1.5 | 4.2 | 679 | 22.5 | 0.79 | 0.97 | 0.55 |

Lower adjacent-bar similarity means more bar-to-bar change. "Corr. below 150 Hz" near 1 means
the low end is mono, as rule M1 intends; in v1 it was not, because of the master-stage bug
described in the diary.

## What changed musically

**A. Rivet Logic.** v1 had a sound plan but a static form: one bass cell for 40 bars, a hook
that doubled the bass's attacks, one fill, and a lift melody that never returned. In v3:
- the hook interlocks in the bass's gaps, so there are two rhythmic voices instead of one;
- every section has its own kick map and hat density;
- the bridge is a valley that the return lands out of;
- the final is the one place everything plays, with the lift's melody back over the hook;
- the mix moved from a dark low end (centroid 325 Hz) to audible mids (688 Hz; mids and presence
  from 5% to 15%).

**B. Lantern Arithmetic.** v1 had the right machinery (incommensurate loops over a moving pedal,
Poisson events, composed treatments) under a drone that buried it. Its level hardly moved (LRA
3.8 LU). In v3:
- the harmony departs further (an A major ninth middle, C withheld for a minute);
- the loops accrete, and the drone leaves a hole;
- a four-note motif is developed by chance fragments;
- the level breathes (LRA 8.3 LU);
- the lows are mono, and the top end exists.

**C. Concrete Tide.** v1 already had the clearest form (dense and space sections alternating),
but its first space section did not open, and its sections arrived fully formed. In v3:
- the dense sections build in stages;
- a one-bar stop precedes the climax;
- the phrygian turn reaches the harmony as well as the bass;
- the fuzz walks across the bar.

## What the numbers do not say

None of these measurements is musical quality. The project's one ear-based judgement so far
(Jonathan on A v3) went to a render that breaks the 3 dB limiting rule by 3.5 dB. The rule is now
advisory. `agent/evaluate.py` exists to measure how far the automated listener's scores track his.
