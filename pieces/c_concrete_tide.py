"""C. Concrete Tide: hybrid art-rock instrumental, 88 BPM, D with a contested second
(dorian E against phrygian Eb). Bass carries the tune; percussion is metal, wood and a dropped
snare; dense and spacious sections alternate (R20).
"""
from dataclasses import replace

import numpy as np

from engine import instruments
from engine import song as engine_song
from engine.dsp import (SR, adsr, butter, filt, filt_tv, ks_string, ladder, mtof, n_of, perc,
                        pulse, saw, sine, undb)
from engine.fx import smooth_noise
from engine.rules import Grid, euclid, nudge, rotate, thin, walk
from engine.song import Note, Song, Track
from pieces.common import add_cell, add_hits, add_line, nn, euclid_steps

BPM = 88
SEED = 3313

# Two-bar bass melody (step, length, semitones above D2, velocity). Bar 2 is the answer.
RIFF_1 = [(0, 3, 0, 1.0), (3, 1, 12, 0.75), (4, 2, 10, 0.85), (6, 2, 7, 0.8), (8, 3, 5, 0.9),
          (11, 1, 3, 0.7), (12, 2, 7, 0.85), (14, 2, 10, 0.8)]
RIFF_2_DORIAN = [(0, 4, 0, 1.0), (4, 1, 2, 0.75), (5, 1, 3, 0.8), (6, 4, 7, 0.9),
                 (10, 2, 5, 0.75), (12, 1, 7, 0.8), (13, 1, 10, 0.8), (14, 2, 12, 0.9)]
RIFF_2_PHRYGIAN = [(s, l, 1 if o == 2 else o, v) for s, l, o, v in RIFF_2_DORIAN]
SPACE_BASS = [(0, 10, 0, 0.9), (10, 6, 7, 0.7)]

# Chords per two bars (voicings, MIDI names), stacked fifths and sus shapes (R21)
PROG = [("D3", "A3", "E4"), ("Bb2", "F3", "A3", "E4"), ("C3", "G3", "D4"), ("D3", "A3", "E4")]
POWER_ROOTS = ["D3", "Bb2", "C3", "D3"]

LEAD_D2 = [(0, 0, 32, "A4"), (2, 0, 24, "G4"), (3, 8, 8, "F4"), (4, 0, 32, "E4"),
           (6, 0, 16, "F4"), (7, 0, 16, "Eb4")]
LEAD_D3 = [(0, 0, 24, "D5"), (1, 8, 8, "C5"), (2, 0, 16, "A4"), (3, 0, 8, "Bb4"),
           (3, 8, 8, "C5"), (4, 0, 32, "D5"), (6, 0, 12, "F5"), (6, 12, 4, "E5"),
           (7, 0, 16, "D5"), (8, 0, 24, "A4"), (9, 8, 8, "C5"), (10, 0, 32, "D5")]

# ---- revision 3 material
# R1-compliant riffs: two quarter-beat onsets at most, the rest on the 'e' and 'a'
RIFF_1_V3 = [(0, 3, 0, 1.0), (3, 1, 12, 0.75), (5, 2, 10, 0.85), (7, 1, 7, 0.8), (8, 3, 5, 0.9),
             (11, 1, 3, 0.7), (13, 2, 7, 0.85), (15, 1, 10, 0.8)]
RIFF_2_DOR_V3 = [(0, 3, 0, 1.0), (3, 2, 2, 0.75), (5, 1, 3, 0.8), (6, 4, 7, 0.9),
                 (10, 1, 5, 0.75), (11, 2, 7, 0.8), (13, 1, 10, 0.8), (14, 2, 12, 0.9)]
# the Eb takes the E and the F's place: one long, loud flat second
RIFF_2_PHR_V3 = [(0, 3, 0, 1.0), (3, 3, 1, 0.95), (6, 4, 7, 0.9), (10, 1, 5, 0.75),
                 (11, 2, 7, 0.8), (13, 1, 10, 0.8), (14, 2, 12, 0.9)]
# climax answer: climbs to Eb3 and F3 instead of resolving
RIFF_CLIMB = [(0, 3, 0, 1.0), (3, 1, 12, 0.8), (6, 2, 13, 0.9), (8, 2, 15, 0.95),
              (11, 1, 13, 0.85), (13, 3, 12, 0.9)]
# bII over the pedal: the Eb is in the chords, not only in one bass sixteenth
PROG_PHRYG = [("D3", "A3", "Eb4"), ("Bb2", "F3", "A3", "D4"), ("Eb3", "Bb3", "F4"),
              ("D3", "A3", "Eb4")]
POWER_ROOTS_PHRYG = ["D3", "Bb2", "Eb3", "D3"]
CODA_MAJOR = ("D3", "A3", "E4", "F#4")      # the only major third in the piece (R22)
KICK_D1 = [(0, 1.0), (10, 0.8)]              # fixed kick map per section (R12)
KICK_D2 = [(0, 1.0), (7, 0.55), (10, 0.8)]
KICK_D3 = [(0, 1.0), (3, 0.5), (10, 0.8), (14, 0.55)]
PIPE_SET = {"dor": ["A5", "C6", "D6", "E6"], "phr": ["A5", "Bb5", "D6", "Eb6"]}
# dense 2 lead lands on the Eb over the bII chord, and its last note is the E against the Eb
LEAD_D2_V3 = [(0, 0, 32, "A4"), (2, 0, 24, "G4"), (3, 8, 8, "F4"), (4, 0, 32, "Eb4"),
              (6, 0, 16, "F4"), (7, 0, 16, "E4")]


def fit_cell(cell, length=16):
    """Sort a cell and trim each note so it ends by the next onset."""
    cell = sorted(cell)
    out = []
    for i, (s, ln, o, v) in enumerate(cell):
        end = cell[i + 1][0] if i + 1 < len(cell) else length
        out.append((s, max(1, min(ln, end - s)), o, v))
    return out


def r1_ok(cell):
    """R1: 5-8 onsets, at most two on quarter beats, at least one on an 'a'."""
    return (5 <= len(cell) <= 8 and sum(s % 4 == 0 for s, *_ in cell) <= 2
            and any(s % 4 == 3 for s, *_ in cell))


def mutate_answer(cell, rng, protect=1):
    """One nudge() or thin() (R3, R18). The flat second (offset `protect`) is never removed."""
    for _ in range(30):
        if rng.uniform() < 0.5:
            kept = [c for c in cell if c[2] == protect]
            cand = thin([c for c in cell if c[2] != protect], rng) + kept
        else:
            cand = nudge(cell, rng)
        cand = fit_cell(cand)
        if cand != fit_cell(cell) and r1_ok(cand):
            return cand
    return cell

PATCHES = {
    "bass": {"t60": 3.0, "bright": 0.5, "pick": 0.22, "ring": 0.12, "sub": 0.5, "drive": 1.8,
             "body_lp": 2600, "hp": 35, "release": 0.08},
    "kit": {
        "kick": {"f_start": 120, "f_end": 44, "p_tau": 0.035, "decay": 0.45, "click": 0.2,
                 "drive": 1.5},
        "snare_drop": {"f1": 170, "f2": 300, "decay": 0.4, "body_decay": 0.12, "noise_f": 2600,
                       "drop": 10, "noise": 1.0, "body": 0.9},
        "tom_low": {"decay": 0.6, "sweep": 0.6, "fall": 4},
        "tom_mid": {"decay": 0.5, "sweep": 0.6, "fall": 4},
        "tom_high": {"decay": 0.45, "sweep": 0.6, "fall": 4},
        "shaker": {}, "noise": {"decay": 1.4},
    },
    "wood": {"block_hi": {}, "block_lo": {}},
    "pipe": {"ratios": [1.0, 2.756, 5.404, 8.933], "decay": 1.6, "click": 0.4},
    "plate": {"ratios": [1.0, 1.594, 2.136, 2.296, 2.653, 3.156], "decay": 2.5, "click": 0.2},
    "tide": {"a": 2.5, "d": 1.5, "s": 0.85, "r": 3.5, "voices": 5, "spread": 0.16,
             "cutoff": 1100, "move_oct": 1.2, "move_rate": 0.12, "q": 1.1},
    "choir": {"a": 1.8, "r": 3.0, "s": 0.9, "voices": 4, "spread": 0.1, "cutoff": 2600,
              "move_oct": 0.5, "formant": "a", "formant_morph": 0.9},
    "lead": {"saw": 0.55, "sq": 0.25, "fmix": 0.2, "ratio": 1.0, "index": 1.2, "a": 0.35,
             "s": 0.9, "r": 0.9, "cutoff": 2400, "q": 1.4, "vib": 0.22, "vib_delay": 0.9,
             "vib_rate": 5.0, "glide": 0.18, "drive": 2.0, "move": 0.4},
    "fuzz": {"t60": 3.5, "bright": 0.8, "pick": 0.15, "ring": 0.25, "hp": 90},
}

# ---- revision 5 material: the riff is the hook (R29)
KICK_D1_V5 = [(0, 1.0), (7, 0.6), (10, 0.8)]
KICK_D2_V5 = KICK_D2 + [(14, 0.5)]
# Two-bar guitar riff (step, length, pitch), locked to RIFF_1_V3 / RIFF_2_DOR_V3. "chop" is a
# muted D3+A3 power chord; "2nd" is the contested second (E4 dorian, Eb4 phrygian). Bar A rises
# A-C-D-F onto the second in the bass's gaps; bar B falls C-A onto G3, a swung sixteenth before
# the bass's G, then rubs the second against the bass's octave pop on step 14.
GTR_A = [(0, 2, "chop"), (2, 1, "chop"), (3, 2, "A3"), (6, 3, "C4"), (10, 1, "D4"),
         (11, 2, "F4"), (14, 2, "2nd")]
GTR_B = [(0, 2, "chop"), (2, 1, "chop"), (3, 2, "C4"), (6, 2, "A3"), (9, 4, "G3"),
         (14, 2, "2nd")]
# accent maps, fixed per section (R12)
GTR_ACCENTS = {
    "intro":   {0: 0.95, 2: 0.6, 3: 0.85, 6: 0.75, 9: 0.8, 10: 0.75, 11: 0.85, 14: 0.9},
    "dense 1": {0: 1.0, 2: 0.55, 3: 0.9, 6: 0.7, 9: 0.75, 10: 0.7, 11: 0.9, 14: 0.85},
    "dense 2": {0: 0.9, 2: 0.6, 3: 0.75, 6: 0.75, 9: 0.8, 10: 0.75, 11: 0.8, 14: 1.0},
    "dense 3": {0: 1.0, 2: 0.85, 3: 0.95, 6: 0.85, 9: 0.9, 10: 0.9, 11: 0.95, 14: 1.0},
    "space":   {0: 0.6, 2: 0.4, 3: 0.65, 6: 0.6, 9: 0.65, 10: 0.6, 11: 0.65, 14: 0.75},
    "coda":    {0: 0.85, 2: 0.5, 3: 0.8, 6: 0.65, 9: 0.7, 10: 0.65, 11: 0.8, 14: 0.75},
}


def bass_accent(step):
    """v5 bass accent map (R12): downbeat, then the 'a' sixteenths, everything else low."""
    return 1.0 if step == 0 else (0.9 if step % 4 == 3 else 0.55)


PATCHES_V5 = {
    "guitar": {"t60": 1.6, "bright": 0.7, "pick": 0.35, "ring": 0.06, "release": 0.04,
               "hp": 90, "pick_noise": 0.15},
    "amp": {"drive": 3.5, "asym": 1.6, "hp": 100.0, "peak_hz": 1800.0, "peak_db": 3.0,
            "lp": 4800.0, "out": 0.45},
    "keys": {"release": 0.45, "tune_cents": 2.5, "drift_cents": 8.0, "drift_rate": 0.13,
             "drift_phase": 0.7, "detune_cents": 7.0,
             "partials": [[1.0, 1.0, 2.4], [2.0, 0.3, 0.9], [3.0, 0.1, 0.4], [4.2, 0.12, 0.1]]},
    "grit": {**PATCHES["bass"], "sub": 0.0},
    "tide": {**PATCHES["tide"], "move_oct": 1.6, "move_rate": 0.07},
    "drone": {"a": 3.0, "d": 2.0, "s": 0.9, "r": 4.0, "voices": 3, "spread": 0.24,
              "cutoff": 900, "move_oct": 0.5, "move_rate": 0.05, "q": 0.8, "drift_cents": 5},
    "air": {"a": 2.0, "r": 2.5, "wander_oct": 0.5, "wander_rate": 0.11, "q": 1.5},
    "swell": {"a": 0.005, "r": 0.3, "wander_oct": 1.0, "wander_rate": 0.6, "q": 2.0},
}


def c_guitar(note, patch, rng):
    """Guitar string: the engine's Karplus-Strong pluck plus a short pick transient, scaled by
    velocity. The amp and cabinet are the c_amp insert, so the double-tracked copies share them."""
    y = instruments.pluck(note, patch, rng)
    k = min(len(y), n_of(0.004))
    click = rng.standard_normal(k) * np.linspace(1.0, 0.0, k)
    y[:k] += patch.get("pick_noise", 0.0) * filt(click, "hp", 2500)
    return y * note.vel


def c_keys(note, patch, rng):
    """Clean, detuned, decaying keyboard (R27): paired sine partials a few cents apart, each
    decaying at its own rate, damped on release, with a slow pitch drift that follows the clock
    rather than the tempo (R13) and a small random tuning error per note."""
    drift = patch.get("drift_cents", 8.0) * np.sin(
        2 * np.pi * patch.get("drift_rate", 0.13) * note.t + patch.get("drift_phase", 0.0))
    f = mtof(note.pitch) * 2 ** ((drift + rng.normal(0.0, patch.get("tune_cents", 2.5))) / 1200)
    rel = patch.get("release", 0.45)
    n = n_of(note.dur + rel)
    det = patch.get("detune_cents", 7.0)
    y = np.zeros(n)
    for mult, amp, dec in patch.get("partials", [[1.0, 1.0, 2.4]]):
        env = perc(dec, n, attack=0.003)
        for side in (-1.0, 1.0):
            y += 0.5 * amp * sine(f * mult * 2 ** (side * det / 1200), n) * env
    nd = min(n, n_of(note.dur))
    damp = np.ones(n)
    damp[nd:] = np.exp(-np.arange(n - nd) / SR * 6.9 / rel)
    return y * damp * note.vel * 0.3


def c_amp(x, rng, drive=3.5, asym=1.6, hp=100.0, peak_hz=1800.0, peak_db=3.0, lp=4800.0,
          out=0.45):
    """Guitar amp and cabinet: asymmetric tanh clipping (the positive half driven `asym` times
    harder), then a cabinet band-limit: high-pass, a presence peak and a 12 dB/oct low-pass.
    The input is referred to the track's own peak, so the amount of clipping is set here."""
    u = x / (float(np.max(np.abs(x))) + 1e-12)
    y = np.where(u >= 0, np.tanh(drive * asym * u) / np.tanh(drive * asym),
                 np.tanh(drive * u) / np.tanh(drive))
    y = butter(y, "high", hp, 2)              # also removes the clipper's DC offset
    y = filt(y, "peak", peak_hz, 1.0, peak_db)
    y = butter(y, "low", lp, 2)
    return y * out


# piece-local sounds (used only by revision 5 onward)
instruments.VOICES["c_guitar"] = c_guitar
instruments.VOICES["c_keys"] = c_keys
engine_song.CUSTOM_INSERTS["c_amp"] = c_amp


def lfo_points(period, lo, hi, end, phase=0.0, step=0.25):
    """lp_auto points for a sine sweep between lo and hi Hz (log scale) with the given period."""
    ts = np.arange(0.0, end + step, step)
    f = lo * (hi / lo) ** (0.5 + 0.5 * np.sin(2 * np.pi * ts / period + phase))
    return [(round(float(t), 3), round(float(v), 1)) for t, v in zip(ts, f)]


# ================================================================ revision 6: a new piece in E
# The hook: a one-bar call and a one-bar answer (step, pitch, length, velocity, slide target).
# Its onsets (2, 4, 8, 9, 11, 13) never meet the bass's (0, 3, 6, 7, 10, 12, 14, 15).
CALL6 = [(2, 59, 1, 0.80, None), (4, 62, 2, 1.00, 64), (8, 67, 1, 0.85, None),
         (9, 66, 1, 0.75, None), (11, 64, 1, 0.90, None), (13, 59, 2, 0.80, None)]
ANSWER6 = [(2, 59, 1, 0.80, None), (4, 62, 2, 1.00, 64), (8, 69, 1, 0.95, None),
           (9, 67, 1, 0.80, None), (11, 66, 1, 0.85, None), (13, 64, 3, 0.90, None)]
# the third guitar's diatonic fourths below (E aeolian); F sharp takes the tritone C
FOURTH_BELOW6 = {59: 54, 62: 57, 64: 59, 66: 60, 67: 62, 69: 64}
# bass cells (step, semitones above the root, length, velocity)
BASS_E6 = [(0, 0, 2, 1.0), (3, 0, 1, 0.65), (6, 12, 1, 0.90), (7, 0, 1, 0.60), (10, 3, 2, 0.95),
           (12, 5, 1, 0.75), (14, 7, 1, 0.85), (15, 10, 1, 0.80)]
BASS_CD6 = [(0, 0, 2, 1.0), (3, 0, 1, 0.65), (6, 12, 1, 0.90), (7, 0, 1, 0.60), (10, -5, 2, 0.95),
            (12, 0, 1, 0.75), (14, 7, 1, 0.85), (15, 12, 1, 0.80)]
BASS_ROOT6 = {"E": 28, "C": 36, "D": 38}            # E1, C2, D2
CYCLE6 = ["E", "E", "C", "D"]
PAD_VOICING6 = {"E": ("E3", "B3", "D4", "F#4", "A4"), "C": ("C3", "G3", "B3", "E4"),
                "D": ("D3", "A3", "E4", "F#4")}
CMAJ9_6 = ("C3", "G3", "B3", "D4", "E4")
EM11_6 = ("E3", "B3", "D4", "F#4", "A4")
K_A6 = [(0, 1.0), (10, 0.9)]
K_B6 = [(0, 1.0), (3, 0.55), (10, 0.9), (14, 0.6)]
K_C6 = [(0, 1.0), (3, 0.7), (6, 0.8), (10, 0.95), (14, 0.7)]

PATCHES_V6 = {
    "bass": {"saw": 0.7, "sq": 0.5, "detune_cents": 6.0, "a": 0.002, "d": 0.25, "s": 0.85,
             "r": 0.04, "cut": 180.0, "fenv": 1220.0, "fdecay": 0.09, "res": 0.25, "drive": 2.2,
             "sub": 0.5},
    "guitar": {"t60": 1.2, "bright": 0.8, "pick": 0.2, "pick_noise": 0.2, "release": 0.04,
               "ring": 0.08, "glide": 0.05},
    "fuzz": {"t60": 1.5, "bright": 0.85, "pick": 0.15, "pick_noise": 0.1, "release": 0.04,
             "ring": 0.08, "glide": 0.05},
    "feedback": {"t60": 6.0, "bright": 0.7, "pick": 0.3, "pick_noise": 0.0, "release": 0.4,
                 "ring": 0.45},
    "amp": {"drive": 5.0, "asym": 1.4, "hp": 90.0, "peak_hz": 2200.0, "peak_db": 3.0,
            "lp": 5500.0, "out": 0.45, "di_db": -10.0, "di_lp": 6000.0},
    "keys": {**PATCHES_V5["keys"], "drift_cents": 5.0, "release": 0.6, "glide": 0.09},
    "pad": {"a": 1.2, "d": 1.5, "s": 0.9, "r": 2.5, "voices": 6, "spread": 0.2, "cutoff": 3500,
            "move_oct": 0.25, "move_rate": 0.09, "q": 0.7, "drift_cents": 4},
    "strings": {"a": 0.8, "d": 1.0, "s": 0.9, "r": 2.5, "voices": 6, "spread": 0.06,
                "cutoff": 3000, "move_oct": 0.3, "move_rate": 0.07, "q": 0.8, "drift_cents": 3},
    "drone": {"a": 2.0, "d": 2.0, "s": 0.9, "r": 3.0, "voices": 3, "spread": 0.08,
              "cutoff": 2500, "move_oct": 0.3, "move_rate": 0.05, "q": 0.8, "drift_cents": 5},
    "bed": {"a": 1.5, "r": 2.0, "wander_oct": 0.25, "wander_rate": 0.09, "q": 2.0},
    "kit": {
        "kick": {"f_start": 160, "f_end": 55, "p_tau": 0.028, "decay": 0.25, "click": 0.4,
                 "drive": 1.6},
        "snare_drop": {"f1": 190, "f2": 330, "decay": 0.3, "body_decay": 0.1, "noise_f": 2500,
                       "drop": 5, "noise": 0.5, "body": 1.0},
        "snare": {"f1": 190, "f2": 330, "decay": 0.12, "body_decay": 0.01, "noise_f": 4000,
                  "noise": 1.0, "body": 0.0},
        "clap": {"f": 1500, "tail": 0.14},
        "hat": {"decay": 0.035, "tune": 1.7, "bp": 9800},
        "tom_low": {"decay": 0.35, "sweep": 0.6, "fall": 4},
        "tom_mid": {"decay": 0.32, "sweep": 0.6, "fall": 4},
        "tom_high": {"decay": 0.3, "sweep": 0.6, "fall": 4},
        "noise": {"decay": 0.9},
    },
    "heart": {"tom_low": {"decay": 0.22, "sweep": 0.3, "fall": 2}},
    "rev_snare": {"snare_drop": {"f1": 190, "f2": 330, "decay": 0.9, "body_decay": 0.3,
                                 "noise_f": 3500, "drop": 7, "noise": 1.0, "body": 0.6}},
    "sheet": {"ratios": [1.0, 1.483, 1.932, 2.546, 3.17], "decay": 0.4, "click": 0.1},
    "anvil": {"ratios": [1.0, 1.18, 1.41], "decay": 0.12, "click": 0.2},
    "tick": {"ratios": [1.0, 1.13, 1.29], "decay": 0.04, "click": 0.1},
}


# ---- revision 7: a thicker hook, louder backgrounds; nothing in the v6 score moves
PATCHES_V7 = {
    # an organ-like body under the hook: drawbar-style partials with long decays, so it holds
    # through each gate instead of decaying like a piano; +-7 cents keeps the pair's beating
    # under 4 Hz on the hook's fundamentals
    "body": {**PATCHES_V6["keys"], "detune_cents": 7.0, "drift_cents": 3.0, "release": 0.12,
             "glide": 0.05,
             "partials": [[1.0, 1.0, 8.0], [2.0, 0.5, 8.0], [3.0, 0.25, 6.0], [4.0, 0.15, 4.0]]},
}
V7_NAMES = ["gtr_c_uni", "gtr_l_uni", "gtr_r_uni", "gtr_oct", "hook_body"]


def _slide(y, semis, glide_s, n_out):
    """Read y at a rate that rises (or falls) by `semis` over `glide_s`, then holds: a slide
    that starts at the attack. Returns n_out samples."""
    r = np.full(n_out, 2.0 ** (semis / 12))
    k = min(n_out, n_of(glide_s))
    r[:k] = 2.0 ** (np.linspace(0.0, semis, k) / 12)
    pos = np.cumsum(r) - r[0]
    return np.interp(pos, np.arange(len(y)), y, right=0.0)


def c_guitar_slide(note, patch, rng):
    """Karplus-Strong string with a pick transient. With `slide_to`, the string is rendered at
    the start pitch and resampled to reach the target over the patch's glide. `feedback` adds
    a sustained harmonic tone with a slow pitch drift (no amplitude modulation); `swell` rises
    from -60 dB to full over that many seconds."""
    p = note.params
    f = float(mtof(note.pitch))
    t60 = p.get("t60", patch.get("t60", 1.2))
    bright = patch.get("bright", 0.8) * (0.6 + 0.4 * note.vel)
    ring = note.dur + patch.get("ring", 0.08)
    n_out = n_of(ring + 0.05)
    semis = float(p["slide_to"] - note.pitch) if p.get("slide_to") is not None else 0.0
    src = (ring + 0.05) * max(1.0, 2.0 ** (semis / 12)) + 0.05
    y = ks_string(f, src, t60, bright, patch.get("pick", 0.2), rng=rng)
    k = min(len(y), n_of(0.004))
    click = rng.standard_normal(k) * np.linspace(1.0, 0.0, k)
    y[:k] += patch.get("pick_noise", 0.0) * filt(click, "hp", 2500)
    y = _slide(y, semis, patch.get("glide", 0.05), n_out) if semis else y[:n_out]
    tt = np.arange(len(y)) / SR
    if p.get("feedback"):
        vib = 2.0 ** (4.0 / 1200 * np.sin(2 * np.pi * 0.23 * tt))
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        y = 0.4 * y + 0.6 * (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph))
    if p.get("swell"):
        y = y * 10.0 ** (3.0 * (np.clip(tt / p["swell"], 0.0, 1.0) - 1.0))
    env = adsr(note.dur, 0.001, 1.0, 1.0, patch.get("release", 0.04))
    m = min(len(env), len(y))
    return y[:m] * env[:m] * note.vel


def c_bass6(note, patch, rng):
    """Saw plus square through a 24 dB low-pass with its own envelope and a drive stage, plus a
    separate sub sine that bypasses the filter (sub_oct 0: at the fundamental; -1: an octave
    below). `dive_to` falls to that pitch over the last `dive_s` of the gate; `sweep` opens the
    base cutoff across the note; `sub_only` keeps just the sine."""
    p = note.params
    env = adsr(note.dur, p.get("a", patch.get("a", 0.002)), patch.get("d", 0.25),
               patch.get("s", 0.85), patch.get("r", 0.04))
    n = len(env)
    fp = np.full(n, float(mtof(note.pitch)))
    if p.get("dive_to") is not None:
        k = min(n, n_of(p.get("dive_s", 0.3)))
        a = max(0, min(n - k, n_of(note.dur) - k))
        fp[a:a + k] = mtof(np.linspace(note.pitch, p["dive_to"], k))
        fp[a + k:] = float(mtof(p["dive_to"]))
    sub = patch.get("sub", 0.708) * sine(fp * 2.0 ** p.get("sub_oct", 0), n)
    if p.get("sub_only"):
        return sub * env * note.vel
    det = 2.0 ** (patch.get("detune_cents", 6.0) / 1200)
    x = patch.get("saw", 0.7) * saw(fp * det, n) + patch.get("sq", 0.5) * pulse(fp / det, n, 0.5)
    c0 = p.get("cut", patch.get("cut", 180.0))
    if p.get("sweep"):
        a0, a1 = p["sweep"]
        c0 = a0 * (a1 / a0) ** (np.arange(n) / n)
    cut = c0 + p.get("fenv", patch.get("fenv", 1220.0)) * perc(patch.get("fdecay", 0.09), n) \
        * (0.4 + 0.6 * note.vel)
    y = ladder(x * 0.6, cut, patch.get("res", 0.25), 1.0)
    dr = patch.get("drive", 2.2)
    y = np.tanh(dr * y) / np.tanh(dr)
    return (y + sub) * env * note.vel


def c_pad6(note, patch, rng):
    """Detuned saw ensemble without periodic beating. The engine pad spaces its voices evenly
    and starts them in phase, so every adjacent pair beats at one rate: an audible tremolo on
    held notes. Here the detune offsets are irregular, each voice starts at a random phase and
    drifts on its own slow curve, so the beating is an aperiodic shimmer. A slow random
    low-pass movement keeps the sound moving (R13)."""
    f = float(mtof(note.pitch))
    env = adsr(note.dur, patch.get("a", 1.2), patch.get("d", 1.0), patch.get("s", 0.8),
               patch.get("r", 2.0))
    n = len(env)
    voices = int(patch.get("voices", 6))
    spread = patch.get("spread", 0.2)
    base = np.linspace(-0.5, 0.5, voices)
    if voices > 2:
        base[1:-1] += rng.uniform(-0.3, 0.3, voices - 2) / (voices - 1)
    x = np.zeros(n)
    for k in range(voices):
        drift = smooth_noise(n, patch.get("drift_rate", 0.2), rng) * patch.get("drift_cents", 4)
        fk = f * 2.0 ** (base[k] * spread / 12 + drift / 1200)
        x += saw(fk, n, rng.uniform()) / voices
    move = smooth_noise(n, patch.get("move_rate", 0.1), rng)
    cut = patch.get("cutoff", 3500) * 2.0 ** (patch.get("move_oct", 0.25) * move)
    y = filt_tv(x, "lp", cut, patch.get("q", 0.7), block=256)
    return y * env * 0.5


def c_keys_slide(note, patch, rng):
    """The c_keys voice, with an optional slide (rendered long, then resampled)."""
    to = note.params.get("slide_to")
    if to is None:
        return c_keys(note, patch, rng)
    semis = float(to - note.pitch)
    longer = replace(note, dur=note.dur * 2.0 ** (max(semis, 0.0) / 12) + 0.1)
    y = c_keys(longer, patch, rng)
    return _slide(y, semis, patch.get("glide", 0.09), n_of(note.dur + patch.get("release", 0.45)))


def c_amp_di(x, rng, di_db=-10.0, di_lp=6000.0, **amp):
    """c_amp, plus a parallel clean DI copy (low-passed) set `di_db` under the amp's RMS, so
    the pitch reads through the dirt (R24). di_db None: amp only."""
    wet = c_amp(x, rng, **amp)
    if di_db is None:
        return wet
    di = butter(x, "low", di_lp, 2)
    rw, rd = float(np.sqrt(np.mean(wet ** 2))), float(np.sqrt(np.mean(di ** 2)))
    if rd < 1e-12:
        return wet
    return wet + di * (rw / rd) * undb(di_db)


def mod_depth_db(x, lo=4.0, hi=20.0, win_s=8.0):
    """Peak-to-peak depth (dB) of the strongest periodic amplitude modulation between lo and hi
    Hz: 20 ms RMS frames every 10 ms, in dB, Hann-windowed 8 s spans where the signal is active
    throughout, strongest spectral line in the band."""
    m = np.asarray(x if x.ndim == 1 else x.mean(axis=0), dtype=float)
    hop, fw = n_of(0.01), n_of(0.02)
    nf = (len(m) - fw) // hop
    fps = SR / hop
    w = int(round(win_s * fps))
    if nf < w:
        return 0.0
    cs = np.concatenate([[0.0], np.cumsum(m * m)])
    st = np.arange(nf) * hop
    lv = 10 * np.log10(np.maximum((cs[st + fw] - cs[st]) / fw, 0.0) + 1e-20)
    act = lv > lv.max() - 30.0
    win = np.hanning(w)
    f = np.fft.rfftfreq(w, 1.0 / fps)
    band = (f >= lo) & (f <= hi)
    worst = 0.0
    for a in range(0, nf - w + 1, w // 2):
        if act[a:a + w].mean() < 0.95:
            continue
        seg = lv[a:a + w] - lv[a:a + w].mean()
        amp = 2.0 * float(np.abs(np.fft.rfft(seg * win))[band].max()) / win.sum()
        worst = max(worst, 2.0 * amp)
    return worst


def c_modcheck(x, rng, label="", limit_db=1.5):
    """Guard against stutter or tremolo on sustained layers: the render fails if this stem's
    4-20 Hz amplitude modulation is deeper than limit_db. The signal passes unchanged."""
    d = mod_depth_db(x)
    print(f"modcheck {label}: {d:.2f} dB (limit {limit_db})")
    if d > limit_db:
        raise RuntimeError(f"modulation check failed on {label}: {d:.2f} dB > {limit_db} dB")
    return x


# piece-local sounds (used only by revision 6 onward)
# ---- revision 8 (variation 1 of 3, built on v7 only): the riff as sampled house-piano stabs
PATCHES_V8 = {
    # short release so the stabs clip at the gate; the harder layers for the brighter recordings
    "piano": {"layer_bias": 0.4, "release": 0.07, "max_len": 1.5, "bright": 3.0, "body": 1.0,
              "tune": 0.0},
}
V8_NAMES = ["piano", "piano_b", "piano_wall", "piano_oct_b", "piano_oct_wall", "piano_stop",
            "piano_echo", "piano_di"]
# every hook guitar and hook layer leaves the rev-8 render (their tracks stay in the list, empty,
# so no later track's noise stream moves); the wall fuzz stays as the one quiet guitar
V8_DROPPED = ["gtr_c", "gtr_c_uni", "gtr_l", "gtr_r", "gtr_l_uni", "gtr_r_uni", "gtr_oct",
              "gtr_4th", "hook_body", "gtr_echo", "gtr_di"]
LH_ROOT8 = {"E": 52, "C": 48, "D": 50}               # left hand: E3, C3, D3
GRACE_S = 0.05                                       # v7's 50 ms glide, as a grace note
# piano gains per section, set by measurement (render_and_measure)
PIANO_DB8 = {"A": 2.5, "B": 1.5, "wall": 2.5, "stop": -5.0, "echo": -3.0, "coda": -8.0}

# ---- revision 9 (variation 2 of 3, built on v7 only): same rhythm, new shape
PATCHES_V9 = {
    # the harder, brighter recorded layers; a very short damper so every stab is clipped
    "piano": {"layer_bias": 0.6, "release": 0.06, "max_len": 1.5, "bright": 3.0, "body": 0.0,
              "tune": 0.0},
}
# The hook keeps v6's onsets (2, 4, 8, 9, 11, 13), all in the bass's gaps, with a new contour:
# (step, hand, right-hand top pitch, length, velocity). The right hand calls high (B4, the
# D-to-E chord move on the snare, then an octave-drop repeat of the peak: G5-G4 in the call,
# A5-A4 in the answer); the left hand answers lower on 11 and 13 with the chords in LH9.
CALL9 = [(2, "R", 71, 1, 0.80), (4, "R", 74, 2, 1.00), (8, "R", 79, 1, 0.90),
         (9, "R", 67, 1, 0.80), (11, "L", None, 1, 0.85), (13, "L", None, 2, 0.80)]
ANSWER9 = [(2, "R", 71, 1, 0.80), (4, "R", 74, 2, 1.00), (8, "R", 81, 1, 0.95),
           (9, "R", 69, 1, 0.80), (11, "L", None, 1, 0.85), (13, "L", None, 3, 0.90)]
# step 4: D5 over A4, then 50 ms later E5 over B4 (parallel fourths in place of the slide)
MOVE9 = ((74, 69), (76, 71))
GRACE_V9 = 0.65                                      # the grace dyad, lighter than its landing
# left-hand chords per harmony; nothing under C3, no third of E. The call ends on B (a
# question), the answer on E.
LH9 = {"call": {"E": {11: (52, 57, 62), 13: (52, 59)},
                "C": {11: (48, 55, 62), 13: (48, 55, 59)},
                "D": {11: (50, 57, 62), 13: (50, 57, 59)}},
       "answer": {"E": {11: (54, 57, 62), 13: (52, 59, 64)},
                  "C": {11: (48, 55, 62), 13: (48, 55, 64)},
                  "D": {11: (50, 57, 62), 13: (50, 57, 64)}}}
# one track per section and per hand, so each is measured on its own
V9_NAMES = ["p9_a_rh", "p9_a_lo", "p9_a_lh", "p9_echo", "p9_b_rh", "p9_b_lo", "p9_b_hi",
            "p9_b_lh", "p9_w_rh", "p9_w_lo", "p9_w_hi", "p9_w_lh", "p9_stop", "p9_coda_rh",
            "p9_coda_lh"]
# piano gains per section and hand, set by measurement (render_and_measure); the octave-below
# stack sits 6 dB under its right hand, the octave-above stack 8 dB under
PIANO_DB9 = {"A_rh": 4.5, "A_lh": -2.0, "B_rh": 3.5, "B_lh": -3.0, "wall_rh": 4.5,
             "wall_lh": -2.0, "stop": -9.0, "echo": -4.0, "coda_rh": -4.0, "coda_lh": -6.0}

# ---- revision 10 (variation 3 of 3, built on v7 only): a new house piano vamp
PATCHES_V10 = {
    # the hardest, brightest recorded layers, a very short damper, no low body
    "piano": {"layer_bias": 0.85, "release": 0.05, "max_len": 1.5, "bright": 3.5, "body": 0.0,
              "tune": 0.0},
}
# Two-bar call and answer of off-beat stabs (step, top pitch, length, velocity, grace-from).
# Steps 2, 5, 9, 11 and 13 never meet the bass (0, 3, 6, 7, 10, 12, 14, 15), the snare (4, 12)
# or any kick map. The call ends on B (a question), with the D-to-E grace on step 9; the answer
# climbs B, D, E to v7's G peak and resolves on a held E.
# (draft 4 struck steps 5 and 13 softer, next to the kicks, to lower the limiter: the piano lost
# energy, the master gain rose and the drum transients hit the limiter harder, 5.2 against
# 5.1 dB, so the draft-3 velocities stand)
CALL10 = [(2, 71, 1, 0.80, None), (5, 74, 1, 0.90, None), (9, 76, 2, 1.00, 74),
          (11, 74, 1, 0.85, None), (13, 71, 1, 0.80, None)]
ANSWER10 = [(2, 71, 1, 0.80, None), (5, 74, 1, 0.90, None), (9, 76, 1, 0.90, None),
            (11, 79, 1, 1.00, None), (13, 76, 3, 0.90, None)]
GRACE_V10 = 0.65                                     # the D chord, lighter than its landing
STAB_FLOOR10 = 59                                    # no stab note below B3
# the left hand's root and fifth, at C4 or above, on steps 2 and 9 (B and the wall)
LH10 = {"E": (64, 71), "C": (60, 67), "D": (62, 69)}
# the space's grace dyad: D5 over A4, then E5 over B4
MOVE10 = ((74, 69), (76, 71))
V10_NAMES = ["p10_a", "p10_a_oct", "p10_echo", "p10_b", "p10_b_oct", "p10_b_hi", "p10_b_lh",
             "p10_w", "p10_w_oct", "p10_w_hi", "p10_w_lh", "p10_stop", "p10_coda"]
# piano gains per section, set by measurement (render_and_measure); the octave-up stack sits
# 4 dB under its stabs, the two-octave sparkle 12 dB under, the left hand 6 dB under
PIANO_DB10 = {"A": 1.5, "B": 1.0, "wall": 2.0, "stop": -9.0, "echo": -4.0, "coda": -4.0}
# fader offsets of the stacks under their stabs: draft 1 (-4 and -12) measured the octave-up
# stack 7 dB and the sparkle 14 dB under the three-note stabs in the mid band, so they come up
OCT_DB10, HI_DB10 = -2.0, -9.0


def stab_voicing(top, pcs, n_inner):
    """Close voicing under a riff note: the nearest chord tones strictly below it and within an
    octave, skipping any tone a minor second under the voice above it."""
    out, prev = [], top
    for c in range(top - 1, top - 12, -1):
        if len(out) == n_inner:
            break
        if c % 12 in pcs and prev - c != 1:
            out.append(c)
            prev = c
    return out


def c_piano(note, patch, rng):
    """The sampled grand piano, with per-note release and max_len overrides (the stop chord
    rings; the stabs are clipped)."""
    from engine import sampler
    pt = {**patch, **{k: note.params[k] for k in ("release", "max_len") if k in note.params}}
    return sampler.piano(note, pt, rng)


instruments.VOICES["c_piano"] = c_piano
instruments.VOICES["c_guitar_slide"] = c_guitar_slide
instruments.VOICES["c_bass6"] = c_bass6
instruments.VOICES["c_keys_slide"] = c_keys_slide
instruments.VOICES["c_pad6"] = c_pad6
engine_song.CUSTOM_INSERTS["c_amp_di"] = c_amp_di
engine_song.CUSTOM_INSERTS["c_modcheck"] = c_modcheck


def build(rev=1):
    # the realisation is fixed to revision 1 so later revisions change only what they say
    rng = np.random.default_rng([SEED, 1])
    g = Grid(BPM, swing=0.52)
    sec = {"intro": (0, 4), "dense 1": (4, 12), "space 1": (12, 16), "dense 2": (16, 24),
           "space 2": (24, 30), "dense 3": (30, 42), "coda": (42, 50)}
    n_bars = 50
    D2 = nn("D2")
    bass, kit, wood, pipe, plate, tide, choir, lead, fuzz = ([] for _ in range(9))

    def chord_for(b):
        return PROG[(b // 2) % 4]

    def perc_dense(b, toms=False, full=False):
        add_hits(kit, g, b, [(0, 1.0), (10, 0.8)] + ([(3, 0.5)] if full else []), 36)
        add_hits(kit, g, b, [(8, 1.0)], 40)                       # half-time dropped snare
        add_hits(wood, g, b, [(s, 0.6 + 0.3 * (s % 4 == 2)) for s in
                              euclid_steps(euclid(5, 16, 2))], 76 if b % 2 else 77,
                 jitter=0.003, rng=rng)
        for s in euclid_steps(euclid(3, 8)) + [x + 8 for x in euclid_steps(euclid(3, 8, 1))]:
            add_cell(pipe, g, b, [(s, 4, 0, 0.7)], nn("A5"))
        add_hits(kit, g, b, [(s, 0.35) for s in euclid_steps(euclid(7, 16))], 70,
                 jitter=0.004, rng=rng)
        if toms:
            for k, s in enumerate(euclid_steps(euclid(3, 16, 3))):
                add_hits(kit, g, b, [(s, 0.7)], [48, 45, 41][k % 3])

    v3 = rev >= 3
    v4 = rev >= 4
    shaker_pat = euclid(4, 11)
    hat, snare_fx = [], []
    rng3 = np.random.default_rng([SEED, 3])         # dense-2 mutation path (R3, R18)
    rngp = np.random.default_rng([SEED, 3, 1])      # pipe pitch walk, hat jitter (R17)

    def mode_v3(b):
        """Which second rules a bar: dorian E or phrygian Eb."""
        if sec["dense 2"][0] <= b < sec["dense 2"][1]:
            return "phr"
        if 34 <= b < 38 or 40 <= b < 42:              # dense 3 trades every 4, then every 2 bars
            return "phr"
        return "dor"

    def chord_v3(b):
        return (PROG_PHRYG if mode_v3(b) == "phr" else PROG)[(b // 2) % 4]

    pipe_pat = euclid(3, 7)
    pipe_prev = [nn("A5")]

    def pipe_global(b, vel=0.7):
        # 3 in 7 on a global sixteenth counter: realigns with the bar only every 7 bars (R16)
        pset = [nn(x) for x in PIPE_SET[mode_v3(b)]]
        for s in range(16):
            if pipe_pat[(b * 16 + s) % 7]:
                p = walk(pset, pipe_prev[0], 1, rngp)[0]
                pipe_prev[0] = p
                add_cell(pipe, g, b, [(s, 4, 0, vel * (1.0 if s % 4 == 0 else 0.85))], p,
                         params={"decay": 0.8})

    def perc_v3(b, kicks, snare=True, wood_on=True, shaker=True, toms=False, hats=False,
                snare_dst=None):
        if kicks:
            add_hits(kit, g, b, kicks, 36)
        if snare:
            add_hits(kit if snare_dst is None else snare_dst, g, b, [(8, 1.0)], 40)
        if wood_on:
            add_hits(wood, g, b, [(s, 0.6 + 0.3 * (s % 4 == 2)) for s in
                                  euclid_steps(euclid(5, 16, 2))], 76 if b % 2 else 77,
                     jitter=0.003, rng=rng)
        if shaker and v4:
            # 4 in 11 on a global sixteenth counter: drifts against the 16-step kick and wood
            # and only realigns with the bar every 11 bars (R8, R16)
            add_hits(kit, g, b, [(s, 0.35) for s in range(16) if shaker_pat[(b * 16 + s) % 11]],
                     70, jitter=0.004, rng=rng)
        elif shaker:
            add_hits(kit, g, b, [(s, 0.35) for s in euclid_steps(euclid(7, 16))], 70,
                     jitter=0.004, rng=rng)
        if hats:
            add_hits(hat, g, b, [(s, 0.25 + 0.05 * (s % 4 == 2)) for s in
                                 euclid_steps(euclid(9, 16))], 42, jitter=0.002, rng=rngp)
        if toms:
            for k, s in enumerate(euclid_steps(euclid(3, 16, 3))):
                add_hits(kit, g, b, [(s, 0.7)], [48, 45, 41][k % 3])

    # ---------------------------------------------------------------- intro
    for b in range(*sec["intro"]):
        for s in euclid_steps(euclid(3, 8)):
            add_cell(pipe, g, b, [(s * 2, 4, 0, 0.5 + 0.1 * b)], nn("A5"))
        add_hits(kit, g, b, [(s, 0.25) for s in euclid_steps(euclid(5, 16))], 70)
    for p in PROG[0]:
        add_cell(tide, g, 0, [(0, 64, 0, 0.6)], nn(p), gate=1.0)
    add_hits(kit, g, 3, [(12, 0.7)], 49)
    if v4:
        # composed join into dense 1: the bar-29 device, a reversed tide swell under the crash
        add_cell(tide, g, 3, [(0, 16, 0, 0.7)], nn("D4"), gate=1.0, params={"reverse": True})

    # ---------------------------------------------------------------- dense 1
    for b in range(*sec["dense 1"]):
        if v3:
            # staggered entry: bass and kick, then the dropped snare, wood and shaker, pipe
            add_cell(bass, g, b, RIFF_1_V3 if b % 2 == 0 else RIFF_2_DOR_V3, D2, gate=0.9)
            # v4: the wood blocks enter with the kick, so the first groove is not a bare kit
            perc_v3(b, KICK_D1, snare=b >= 6, wood_on=b >= (4 if v4 else 8), shaker=b >= 8)
            if b >= 10:
                pipe_global(b)
        else:
            add_cell(bass, g, b, RIFF_1 if b % 2 == 0 else RIFF_2_DORIAN, D2, gate=0.9)
            perc_dense(b)
        if b % 2 == 0:
            for p in chord_for(b):
                add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)

    # ---------------------------------------------------------------- space 1 (R20)
    for b in range(*sec["space 1"]):
        add_cell(bass, g, b, SPACE_BASS, D2, gate=0.95)
        add_cell(plate, g, b, [(6, 8, 0, 0.6)], nn("D4") + (b % 2) * 5)
        if b % 2 == 0:
            for p in PROG[1]:
                add_cell(tide, g, b, [(0, 32, 0, 0.55)], nn(p), gate=1.0)

    # ---------------------------------------------------------------- dense 2 (phrygian turn)
    if v3:
        ans = list(RIFF_2_PHR_V3)
        for b in range(*sec["dense 2"]):
            if b % 2 == 1 and b > sec["dense 2"][0] + 1:
                ans = mutate_answer(ans, rng3)              # cumulative drift, one change each
            add_cell(bass, g, b, RIFF_1_V3 if b % 2 == 0 else ans, D2, gate=0.9)
            perc_v3(b, KICK_D2, toms=(b % 4 == 3), hats=True, snare_dst=snare_fx)
            pipe_global(b)
            if b % 2 == 0:
                for p in chord_v3(b):
                    add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
        add_line(lead, g, sec["dense 2"][0], LEAD_D2_V3, gate=0.97)
    else:
        for b in range(*sec["dense 2"]):
            add_cell(bass, g, b, RIFF_1 if b % 2 == 0 else RIFF_2_PHRYGIAN, D2, gate=0.9)
            perc_dense(b, toms=(b % 4 == 3))
            if b % 2 == 0:
                for p in chord_for(b):
                    add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
        add_line(lead, g, sec["dense 2"][0], LEAD_D2, gate=0.97)
    if v4:
        # the dense sections dissolve into plate and echo tails instead of stopping: D at the
        # end of dense 1, E (against the phrygian Eb, and foreshadowing space 2's E) at dense 2
        add_cell(plate, g, sec["dense 1"][1] - 1, [(12, 8, 0, 0.55)], nn("D4"))
        add_cell(plate, g, sec["dense 2"][1] - 1, [(12, 8, 0, 0.55)], nn("E4"))
        # into dense 2: crash on step 12, and the swell is added after the space-1 pad rest
        add_hits(kit, g, sec["dense 2"][0] - 1, [(12, 0.8)], 49)

    # ---------------------------------------------------------------- space 2: no bass
    for b in range(*sec["space 2"]):
        if b % 2 == 0:
            for p in ("Bb2", "F3", "C4", "E4"):          # Bb lydian cluster, no D anywhere
                add_cell(choir, g, b, [(0, 32, 0, 0.55)], nn(p), gate=1.0)
        add_cell(plate, g, b, [(0, 8, 0, 0.45)], nn("E4"))
    add_line(lead, g, sec["space 2"][0], [(0, 4, 28, "E5"), (2, 0, 24, "F5"), (4, 0, 28, "E5")],
             gate=0.98, vel=0.6)
    b = sec["space 2"][1] - 1
    add_hits(kit, g, b, [(12, 0.9)], 49)
    add_cell(tide, g, b, [(0, 16, 0, 0.8)], nn("D4"), gate=1.0,
             params={"reverse": True})

    # ---------------------------------------------------------------- dense 3: climax
    STOP = 37
    if v3:
        # three 4-bar stages; a one-bar stop at 37 (R19) makes bar 38 the climax downbeat
        sv = 0.7 if v4 else 0.85                            # v4: the fuzz steps back
        stabs = [(0, 5, 0, sv), (6, 3, 0, sv), (10, 6, 0, sv)]
        for b in range(*sec["dense 3"]):
            if b == STOP:
                add_cell(plate, g, b, [(0, 8, 0, 0.6)], nn("D4"))
                continue
            m = mode_v3(b)
            if b % 2 == 0:
                cell = RIFF_1_V3
            elif b >= 38:
                cell = RIFF_CLIMB
            else:
                cell = RIFF_2_DOR_V3 if m == "dor" else RIFF_2_PHR_V3
            add_cell(bass, g, b, cell, D2, gate=0.9)
            if v4 and b >= 38:
                # the climax toms replace the wood and shaker rather than stacking on them
                perc_v3(b, KICK_D3, wood_on=False, shaker=False, toms=True, hats=True)
            else:
                perc_v3(b, KICK_D3, toms=(b >= 38 or b % 2 == 1), hats=True)
            if b >= 34:
                pipe_global(b)
            root = nn((POWER_ROOTS_PHRYG if m == "phr" else POWER_ROOTS)[(b // 2) % 4])
            if b < 34:
                pat = []
            elif b < 36:
                pat = stabs[:1]
            elif b < 38:
                pat = [stabs[0], stabs[2]]
            else:
                # walks one sixteenth per bar (R5), starting one late so it never doubles the
                # kick on the climax downbeat
                pat = rotate(stabs, b - 37)
            for s, ln, _, v in pat:
                for iv in (12, 19, 24):                     # an octave up: the bass owns 70-220 Hz
                    add_cell(fuzz, g, b, [(s, ln, iv, v)], root, gate=0.92, jitter=0.004,
                             rng=rng)
            if b % 2 == 0 and not (v4 and b >= 38):
                ln = 16 if b == STOP - 1 else 32            # the choir releases into the stop
                for p in chord_v3(b):
                    add_cell(choir, g, b, [(0, ln, 0, 0.5)], nn(p) + 12, gate=1.0)
            if v4 and b == 40:
                # one held chord two octaves up, above the lead's D5: the fifth and the flat
                # second only, so it neither doubles the lead nor refills the 300-700 Hz octave
                for p in ("A3", "Eb4"):
                    add_cell(choir, g, b, [(0, 32, 0, 0.4)], nn(p) + 24, gate=1.0)
        # the lead's E5 at bar 36 echoes into the stop; its D5 on bar 37 is left out
        add_line(lead, g, sec["dense 3"][0], [x for x in LEAD_D3 if x[0] != STOP - 30],
                 gate=0.97)
    else:
        for b in range(*sec["dense 3"]):
            add_cell(bass, g, b, RIFF_1 if b % 2 == 0 else
                     (RIFF_2_DORIAN if (b // 2) % 2 == 0 else RIFF_2_PHRYGIAN), D2, gate=0.9)
            perc_dense(b, toms=(b % 2 == 1), full=True)
            root = nn(POWER_ROOTS[(b // 2) % 4])
            for s, ln in ((0, 5), (6, 3), (10, 6)):
                for iv in (0, 7, 12):
                    add_cell(fuzz, g, b, [(s, ln, iv, 0.85)], root, gate=0.92, jitter=0.004,
                             rng=rng)
            if b % 2 == 0:
                for p in chord_for(b):
                    add_cell(choir, g, b, [(0, 32, 0, 0.5)], nn(p) + 12, gate=1.0)
        add_line(lead, g, sec["dense 3"][0], LEAD_D3, gate=0.97)

    # ---------------------------------------------------------------- coda: subtraction
    for i, b in enumerate(range(*sec["coda"])):
        if v3:
            cell = RIFF_1_V3 if b % 2 == 0 else RIFF_2_DOR_V3
            if b >= 46:                                 # the minor third leaves before the F#
                cell = [c for c in cell if c[2] != 3]
        else:
            cell = RIFF_1 if b % 2 == 0 else RIFF_2_DORIAN
        keep = max(2, len(cell) - i)                    # one onset fewer per bar (R3)
        add_cell(bass, g, b, cell[:keep], D2, gate=0.95)
        if i < 4:
            for s in euclid_steps(euclid(3, 8)):
                add_cell(pipe, g, b, [(s * 2, 4, 0, 0.5 - 0.08 * i)], nn("A5"))
        if i % 2 == 0:
            chord = CODA_MAJOR if v3 and b >= 46 else PROG[0]
            for p in chord:
                add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
    add_cell(bass, g, n_bars - 1, [(8, 8, 12, 0.6)], D2, gate=1.0)
    add_cell(plate, g, n_bars - 1, [(8, 8, 0, 0.5)], nn("D4"))

    length = g.bar(n_bars)
    v2 = rev >= 2
    gn = {"kick": -3.0, "wood": -15.0, "pipe": -15.0, "plate": -13.0}
    if v2:
        # space 1 opens up: the pad rests, so bass, metal and echo tails carry it (R20)
        a, b = g.bar(sec["space 1"][0]), g.bar(sec["space 1"][1])
        tide = [n for n in tide if not a - 0.01 <= n.t < b]
        for bb in range(*sec["space 1"]):
            add_cell(pipe, g, bb, [(14, 4, 12, 0.35)], nn("A5"))
        gn = {"kick": -4.0, "wood": -11.5, "pipe": -12.0, "plate": -11.0}
    kit_patch = PATCHES["kit"]
    ins = {"kick": [], "bass": [], "kit": [], "pipe": [], "tide": [], "choir": [],
           "fuzz_eq": [["peak", 1600, 1.0, 3.0], ["hp", 110, 0.7]]}
    if v3:
        kit_patch = {**PATCHES["kit"], "snare_drop": {**PATCHES["kit"]["snare_drop"],
                                                      "noise_f": 4500},
                     "hat": {"decay": 0.04, "tune": 1.6, "bp": 9500}}
        # tape sag into the dense 3 downbeat (R14): the tide's reversed swell dips 70 cents
        t29, t30 = g.bar(29), g.bar(30)
        vs = [(t29, 0.0), (t29 + g.steps(12), -70.0), (t30, 0.0)]
        tt = np.linspace(t29, t30, 4001)
        lag = float(np.mean(1 - 2 ** (np.interp(tt, [p[0] for p in vs], [p[1] for p in vs])
                                      / 1200)) * (t30 - t29))
        for n in tide:                       # the slowed read head lands late: start early
            if n.t >= t29 - 0.01:
                n.t -= lag
        t38, t42 = g.bar(38), g.bar(42)
        # the limiter's worst peaks were bass + kick (+ crash) on re-entry downbeats: shave the
        # transients on those tracks rather than lower the loudness target (M2)
        ins = {"kick": [("softclip", {"shave_db": 3.0})],
               "bass": [("softclip", {"shave_db": 3.0})],
               "kit": [("softclip", {"shave_db": 3.0})],
               # ring modulation at an Eb, composed for the climax only (R15)
               "pipe": [("ringmod", {"freq": 311.13, "mix_points": [
                   (0.0, 0.0), (t38 - 0.02, 0.0), (t38, 0.6), (t42, 0.6), (t42 + 0.4, 0.0)]})],
               "tide": [("eq", {"bands": [["peak", 350, 1.0, -3.0]]}),
                        ("varispeed", {"points": vs})],
               "choir": [("eq", {"bands": [["peak", 350, 1.0, -3.0]]})],
               "fuzz_eq": [["peak", 1600, 1.0, 3.0], ["hp", 220, 0.7]]}
    # revision 4 mix and sound-design values (rev <= 3 keep the values above)
    bass_shelf, lead_patch, pipe_patch = 1.5, PATCHES["lead"], PATCHES["pipe"]
    fuzz_ins = {"gain": 10, "tone": 4200, "out": 0.6}
    fuzz_gain, fuzz_sends = -14.0, {}
    lead_ins = []
    if v4:
        # composed join into dense 2 (R14, R19): a reversed Eb swell, foreshadowing the
        # phrygian turn, sags 40 cents on the tape, as the bar-29 swell does
        t15, t16 = g.bar(sec["dense 2"][0] - 1), g.bar(sec["dense 2"][0])
        add_cell(tide, g, sec["dense 2"][0] - 1, [(0, 16, 0, 0.7)], nn("Eb4"), gate=1.0,
                 params={"reverse": True})
        vs15 = [(t15, 0.0), (t15 + g.steps(12), -40.0), (t16, 0.0)]
        tt = np.linspace(t15, t16, 4001)
        lag15 = float(np.mean(1 - 2 ** (np.interp(tt, [p[0] for p in vs15],
                                                  [p[1] for p in vs15]) / 1200)) * (t16 - t15))
        for n in tide:                       # every later tide note starts early by this lag
            if n.t >= t15 - 0.01:
                n.t -= lag15
        tide_tape = ("tape", {"drive": 0.9, "wow": 0.35, "flutter": 0.1, "hf": 8000.0,
                              "bump_db": 0.0})
        ins["tide"] = [("eq", {"bands": [["peak", 350, 1.0, -3.0]]}),
                       ("varispeed", {"points": vs15 + vs}), tide_tape]
        lead_ins = [("eq", {"bands": [["peak", 400, 1.0, -3.0]]}),
                    ("tape", {"drive": 0.9, "wow": 0.25, "flutter": 0.1, "hf": 9000.0,
                              "bump_db": 0.0})]
        # the fuzz retreats: no 1.6 kHz lift, a 2.8 kHz dip, a 4.5 kHz low-pass, less gain,
        # and a plate and echo to sit in
        ins["fuzz_eq"] = [["peak", 2800, 1.2, -4.0], ["hp", 220, 0.7], ["lp", 4500, 0.7]]
        fuzz_ins = {"gain": 6, "tone": 2800, "out": 0.6}
        fuzz_gain, fuzz_sends = -18.0, {"plate": 0.25, "echo": 0.15}
        bass_shelf = 0.5                     # the punch is the 800 Hz peak and the drive
        gn = {**gn, "kick": -5.0}
        kit_patch = {**kit_patch, "snare_drop": {**kit_patch["snare_drop"], "drop": 16,
                                                 "decay": 0.55}}
        pipe_patch = {**PATCHES["pipe"], "click": 0.15}
    tracks = [
        Track("bass", "pluck", PATCHES["bass"], bass, gain_db=0.0, program=35,
              stereo={"chorus": {"depth_ms": 1.5, "rate": 0.5, "mix": 0.25}},
              inserts=[("comp", {"thresh_db": -14, "ratio": 3, "attack": 0.01, "release": 0.12}),
                       ("eq", {"bands": [["peak", 800, 1.0, 2.5],
                                         ["lowshelf", 100, 0.7, bass_shelf]]})]
              + ins["bass"],
              duck=("kick", 4.0 if v3 else 2.0, 0.08), sends={"room": 0.1}),
        Track("kick", "drums", kit_patch, [n for n in kit if n.pitch == 36], gain_db=gn["kick"],
              inserts=ins["kick"]),
        Track("kit", "drums", kit_patch, [n for n in kit if n.pitch != 36], gain_db=-5.0,
              pan=0.1, sends={"plate": 0.35}, inserts=ins["kit"]),
        Track("wood", "drums", PATCHES["wood"], wood, gain_db=gn["wood"], pan=-0.45,
              sends={"room": 0.3}),
        Track("pipe", "metal", pipe_patch, pipe, gain_db=gn["pipe"], pan=0.5,
              sends={"plate": 0.3, "echo": 0.2}, program=14, inserts=ins["pipe"]),
        Track("plate", "metal", PATCHES["plate"], plate, gain_db=gn["plate"], pan=-0.3,
              sends={"plate": 0.5, "echo": 0.3}, program=112),
        Track("tide", "pad", PATCHES["tide"], tide, gain_db=-12.0,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.18, "mix": 0.7}},
              sends={"hall": 0.45}, program=89, inserts=ins["tide"]),
        Track("choir", "pad", PATCHES["choir"], choir, gain_db=-13.0,
              stereo={"chorus": {"depth_ms": 6.0, "rate": 0.12, "mix": 0.7}},
              sends={"hall": 0.5}, program=52, inserts=ins["choir"]),
        Track("lead", "lead", lead_patch, lead, gain_db=-11.0, pan=0.2,
              sends={"echo": 0.4, "hall": 0.25}, program=81, inserts=lead_ins),
        Track("fuzz", "pluck", PATCHES["fuzz"], fuzz, gain_db=fuzz_gain,
              stereo={"haas": 14}, inserts=[("fuzz", fuzz_ins),
                                            ("eq", {"bands": ins["fuzz_eq"]})],
              sends=fuzz_sends, program=30),
    ]
    if v3:
        tracks += [
            # dry closed hat above 7 kHz (R7, R9): the kit's upper register
            Track("hat", "drums", kit_patch, hat, gain_db=-1.0, pan=-0.2, sends={}),
            # dense 2's dropped snare, sample-held and bit-reduced (R15)
            Track("snare_fx", "drums", kit_patch, snare_fx, gain_db=-5.0, pan=0.1,
                  sends={"plate": 0.35}, inserts=[("decimate", {"hold": 6, "bits": 8})]),
        ]
    buses = {
        "room": {"type": "reverb", "ir": {"decay": 0.8, "predelay": 0.008, "tone": 7000,
                                          "size": 0.6}, "return_db": -8},
        "plate": {"type": "reverb", "ir": {"decay": 2.4, "predelay": 0.02, "tone": 7500},
                  "return_db": -5},
        "hall": {"type": "reverb", "ir": {"decay": 5.0, "predelay": 0.035, "tone": 5500,
                                          "size": 1.8}, "return_db": -4},
        "echo": {"type": "echo", "echo": {"time_s": g.steps(6), "feedback": 0.5, "repeats": 6,
                                          "tone": 3200, "spread": 0.7, "wow": 1.2, "sat": 0.4},
                 "return_db": -6},
    }
    master = {"target_lufs": -14.0, "ceiling": -1.5, "tail": 5.0, "linear_phase_mono": v2,
              "eq": [["hp", 28, 0.7]] + ([["highshelf", 7000, 0.7, 2.0]] if v2 else []),
              "comp": {"thresh_db": -20 if v3 else -18, "ratio": 1.5, "attack": 0.03,
                       "release": 0.25},
              "tape": {"drive": 0.7, "wow": 0.08, "flutter": 0.08, "hiss_db": -78,
                       "hf": 14500, "bump_db": 1.2}}
    if v3:
        master["true_peak"] = True
    sections = [(name, g.bar(a), g.bar(b)) for name, (a, b) in sec.items()]
    notes = {1: "First complete arrangement.",
             2: "Space 1 opens up: the pad rests, leaving the bass, metal and echo tails, with a "
                "high pipe answer each bar. Wood +3.5 dB, pipe +3 dB, plate +2 dB, kick -1 dB, "
                "presence shelf, lows truly mono.",
             3: "The bass riffs are re-placed so that only the downbeat and the snare beat fall "
                "on quarter beats; the rest sit on the 'e' and the 'a'. Dense 1 builds in "
                "two-bar stages: bass and kick, then the dropped snare, then wood and shaker, "
                "then the pipe. Dense 2 turns phrygian in the harmony as well as the bass: the "
                "pads carry a bII Eb chord over the D, the Eb in the answer bar is long and "
                "loud, the answer drifts by one nudge or thinning every two bars, the lead "
                "lands on Eb and ends on the E against it, and the dropped snare is "
                "sample-held. Dense 3 builds in three four-bar stages, with E and Eb trading "
                "every four bars and then every two; the fuzz enters on one stab, then two, "
                "then the full pattern walking a sixteenth per bar, never on the climax "
                "downbeat. A one-bar stop at bar 37 "
                "leaves only the plate and the echo tails, and the climax re-enters on bar 38 "
                "with a climbing bass answer, toms every bar and a ring-modulated pipe. The "
                "tide's reversed swell sags 70 cents into the dense 3 downbeat. The pipe now "
                "runs 3 in 7 across the bar line and walks over the section's pitch set. The "
                "coda's last two chords add F sharp, the piece's only major third. Each dense "
                "section has its own kick map. Fuzz an octave higher with a 220 Hz high-pass, "
                "pads cut at 350 Hz, a dry closed hat in dense 2 and 3, brighter snare noise, "
                "soft clips on the kick, kit and bass, a deeper duck of the bass under the "
                "kick, and a lower master compressor threshold, so the limiter stays under "
                "3 dB on the re-entry downbeats.",
             4: "The fuzz stabs step back, as they were the harsh, dry, staccato synth in dense "
                "3. The 1.6 kHz lift is replaced by a 4 dB dip at 2.8 kHz and a 4.5 kHz "
                "low-pass. The fuzz gain falls from 10 to 6, its tone from 4.2 to 2.8 kHz and its "
                "level by 4 dB, and the stabs are softer. They now sit in the plate and the echo "
                "instead of being bone dry. The climax is thinned by subtraction: the choir "
                "does not return after the bar-37 stop, so the climbing bass, the lead, the "
                "walking fuzz and the ring-modulated pipe carry bars 38 to 41 alone. A single "
                "high choir chord on A and E flat enters at bar 40, two octaves up and above the "
                "lead. The climax toms replace the wood and shaker rather than stacking on them. "
                "The wood blocks now enter with the kick at bar 4, so the first groove is not "
                "a bare kit. The shaker runs 4 in 11 across the bar line, so it drifts against "
                "the kick and the wood. The dropped snare falls 16 semitones over a longer "
                "decay, so the drop is heard, and the pipe's click is softer. Transitions are "
                "composed rather than cut: a reversed tide swell leads into dense 1, and a "
                "reversed E flat swell with a crash and a 40-cent tape sag leads into dense 2. "
                "Dense 1 and dense 2 each end on a plate hit that rings out through the echo. "
                "The bass's low shelf drops from 1.5 to 0.5 dB and the kick by 1 dB, to bring "
                "the limiter back towards 3 dB (M2). The lead gets a 3 dB cut at 400 Hz. The "
                "tide and the lead go through a light tape stage with gentle wow, so the "
                "sustained voices drift and warm up rather than sounding sterile.",
             5: "A larger rewrite around a guitar riff, after Jonathan heard v4 as basic, with no "
                "hook, groove or lushness. The piece now has a hook: a two-bar guitar riff, built "
                "in this file as a plucked string with a pick transient into an asymmetric amp "
                "clip and a band-limited cabinet, locked to the bass riff. Muted D power-chord "
                "chops open it, then A, C, D and F climb onto the contested second (E in "
                "dorian bars, E flat in phrygian bars); the answer bar falls to G and then "
                "rubs the second against the bass's octave pop. It is stated alone, mono and "
                "dry, in intro bars 2 and 3, played by a single guitar left of centre in dense "
                "1, and double-tracked hard left and right in dense 2 and dense 3, the right "
                "copy 9 ms late, 7 cents sharp and with its own timing. In the space sections "
                "it becomes a clean, detuned, decaying keyboard through tape wow: a faint, "
                "echoed ghost in space 1, and close and dry in space 2. In dense 3 the fuzz "
                "doubles only the riff's tail, an octave up. The dense-2 lead is cut to its "
                "last two notes as an answer, and the space-2 and dense-3 lead lines are gone. "
                "The groove: swing rises to 55 per cent, the bass is clipped to a 60 per cent "
                "gate with accents on the downbeat and the 'a' sixteenths, and a fuzzed, "
                "band-limited copy of the bass sits about 10 dB under it so that it bites on "
                "small speakers while the sub stays clean. The dense-1 kick gains a hit on "
                "step 7 and the dense-2 kick a ghost on step 14. For lushness, a bed of three "
                "layers runs under the rhythm section: high breath noise, a detuned D and A "
                "drone on tape, and a bit-crushed copy of the pads. Each has its own slowly "
                "moving low-pass (periods of 13.7, 21.1 and 34.3 seconds), the bed changes "
                "from section to section, and it ducks under the kick with the pads. The tide "
                "pad moves more slowly and further, with a slower chorus. Space 2 is now "
                "intimate: the bed thins to the breath alone, and the keyboard riff and the "
                "choir are close and dry, with no hall. For dynamics, bar 36 strips to the "
                "bass and the riff alone, and bar 37 keeps only the plate, a reversed breath "
                "swell and one muted riff chop. Bar 38 then brings everything in at once as a "
                "wall that holds until bar 42: double guitars, the bass and its grit on the "
                "climbing answer, the full bed, the choir back an octave up, toms and wood, a "
                "noise burst on every downbeat in place of a crash, and a crushed parallel "
                "copy of the kit. The coda keeps the riff for its first two bars, so the piece "
                "ends on its hook. In the mix, the bass's 800 Hz lift falls to 1 dB, the grit "
                "is dipped at 1.8 kHz to leave room for the guitar's cabinet peak, the pads "
                "are cut by 2 dB at 2.5 kHz, the guitars get only a short room and a little "
                "echo, and the master compressor threshold returns to -18 dB.",
             6: "A new piece built from scratch, after a guest rater and Jonathan rejected v5 as generic, "
                "short of bass and weak in the beat. Nothing from v1 to v5 is reused apart from "
                "the sound-building code. It stays at 88 BPM but moves to E. The hook is a "
                "one-bar call and a one-bar answer on distorted guitar. It rises B, E, G, steps "
                "down, then drops to B as a question or settles on E as an answer. Its "
                "signature is a D-to-E slide that lands with the snare on beat two. Every "
                "guitar onset falls in a gap of the bass line, so the two interlock. The guitar "
                "is a new sliding string voice through the asymmetric amp, with a clean DI copy "
                "10 dB under it so the pitch reads through the dirt. The riff is "
                "re-orchestrated, never replaced. It is played by one guitar left of centre in "
                "section A and by the slide alone through a dotted-eighth echo in the first "
                "space. Section B double-tracks it hard left and right. In the bridge a quiet, "
                "detuned keyboard plays it at half speed. The wall double-tracks it with a fuzz "
                "an octave up, and in its last four bars a third guitar adds fourths below. At "
                "the end the clean DI plays it alone, dry. The bass is a new saw-and-square "
                "synth bass at E1, with a filter envelope, a drive stage and a separate sub "
                "sine. It plays an accented sixteenth-note cell of eight onsets that changes "
                "only by subtraction: the ghost leaves every fourth bar, and a pitch dive leads "
                "into each section. It ducks 5 dB under a kick tuned to A1, and every kick "
                "lands on a bass note, so the two hit as one punch. A fuzzed copy of the bass "
                "is added in section B and the wall. The beat is a real backbeat on two and "
                "four. The snare has three layers: a dropped-pitch body, a 3-5 kHz noise crack "
                "and a metal sheet tuned to B flat, the tritone over E. Closed hats run in "
                "sixteenths with the eighths accented, and each section has its own kick map. "
                "The wall adds a clap, a metal tick every three sixteenths, noise bursts in "
                "place of crashes and a crushed parallel drum copy. The form runs: a feedback "
                "intro, the groove, section A, a near-empty space, a wider section B, then a "
                "bridge with no bass or drums. A four-on-the-floor build with a snare roll "
                "follows, then half a bar of silence. Next comes a wall where everything plays "
                "at once, then a hard stop, the clean riff alone and a tape stop. For texture "
                "there are a lush detuned pad on a slow filter sweep, a bit-crushed static bed, "
                "a rust drone on tape that is ring-modulated in the wall, high strings, "
                "reversed guitar swells, and a 50-cent tape sag in the bridge. The guitar now "
                "leads the mix: the bass is low-passed at 1.2 kHz, the grit is dipped at "
                "2.2 kHz, and the textures are set to be heard. No sustained layer ducks, tape "
                "flutter stays at 0.04 or below, and each sustained stem must pass a 4-20 Hz "
                "modulation check, so the stuttering tremolo of v5 cannot return. That check "
                "caught the engine pad itself: its evenly spaced, in-phase detuned voices beat "
                "at a single rate, a wobble of about 2 dB on held notes. So the pads, strings "
                "and drone use a new ensemble voice with irregular detune, random start phases "
                "and a separate drift for each voice, which shimmers without pulsing. The B-flat "
                "ring modulation of the drone in the wall is the one deliberate exception: its "
                "sidebands rasp against E and B by design, so the drone is checked before it. "
                "The first full draft measured the sub at half of all the energy and the limiter "
                "at 6 dB, with the hats, tick and anvil inaudible. So the bass's sub sine now sits "
                "6 dB under its oscillators and a 7 dB low shelf sits below 90 Hz, keeping the "
                "growl and losing the boom. The kick comes down 5 dB and the grit up 10 dB, and "
                "the pads and drone come down 5 to 7 dB, so they sit well under the guitars. "
                "The high percussion and the snare layers come up into hearing. The quiet "
                "sections (intro, space, bridge and coda) come down 5 to 11 dB, so that once "
                "the low end stops taking the loudness, they stay quiet against the wall. The "
                "second draft still had the limiter at 5.8 dB, because the drum layers, the bass "
                "and the stop chord all peak on the same instants. So the kick and snare "
                "transients are shaved harder, and the toms come down 4 dB and leave the "
                "crushed copy. The bar-45 power chord is picked softer, and the wall's "
                "feedback guitar drops 5 dB under the hook doubles, which it had been "
                "covering.",
             7: "A narrow revision of v6, which Jonathan rated 6.5 and heard as a sleeper house tune "
                "circa 1992. He asked for exactly two changes, and nothing else is touched. The "
                "rhythm, the swing, the drums, the bass and its lock with the kick, the hook's notes "
                "and rhythm, and the form are all exactly as in v6. The listening model's comments on "
                "the drums are not acted on, because Jonathan likes the beat. First, the hook was too "
                "thin, so it is thickened the way early-90s house records thickened a lead, without "
                "moving a note or an onset. Every new layer is fed from the same hook phrases, so the "
                "slides and gates are identical. Each hook guitar gets a slightly detuned unison "
                "partner, 9 to 11 cents away and 5 to 6 ms late, on its own timing seed: on the "
                "right of centre in section A, and on each guitar's own side in section B and the "
                "wall. The detune is small enough that the beating stays under 4 Hz, so no "
                "tremolo can return. "
                "Sections A and B gain a lightly driven guitar an octave up, about 13 dB under the "
                "lead. In the wall the existing fuzz already plays that part. Every hook guitar "
                "gets a light chorus. Underneath sits a new organ-like body: sustaining drawbar "
                "partials, softly saturated, band-limited to 200 Hz to 2.8 kHz and put through "
                "tape. It holds through each gate where the plucked string decays, and it sits "
                "about 10 dB under the guitar. The clean DI coda stays dry and single. The "
                "guitars were trimmed by 3 dB to make room for the layers, so the hook group "
                "measures within 0.3 dB of v6 in each section, overall and in the mid band. The "
                "unison partners skip the bar-45 stop chord, where the limiter's peaks stack. "
                "Second, the pads and the drone come up by about 3.6 dB, as Jonathan asked. The "
                "three pads get a 2 dB dip at 2.2 kHz, so they swell around the guitar's cabinet "
                "peak rather than over it. Their sound, sweeps, ring modulation and modulation "
                "checks are unchanged. The limiter's peak reduction falls slightly, from 4.7 to "
                "4.5 dB. It stays above 3 dB because its peaks are the drum and bass transients, "
                "which Jonathan asked to leave alone.",
             8: "Variation 1 of 3 of the hook, built on v7 alone (the next two variations also "
                "start from v7). Jonathan rated v7 7/10. He heard it as a house tune from around "
                "1992, but said the riff was the wrong instrument and lacked texture. He chose a "
                "sampled house piano. In this variation the riff keeps every v7 onset, pitch, "
                "length, gate and velocity, and is played as piano chord stabs. Each riff note is "
                "on top of a close chord built from the pad's own pitches for that bar: the two "
                "nearest chord tones below it, within an octave, with no minor seconds between "
                "neighbouring voices. These inner voices are played a little softer, so the "
                "melody leads. Over E this gives E minor ninth and eleventh shapes that still "
                "withhold the third, as the pads do. On the quarter-beat steps the left hand adds "
                "E3, C3 or D3, which never meets a bass note, so the lock with the kick is "
                "untouched. A piano cannot slide, so the D-to-E slide becomes a grace note. The "
                "chord is struck with D on top, and 50 ms later an E replaces the D while the "
                "inner voices hold. The piano follows the house recipe: a high-pass at 120 Hz, a "
                "dip at 300 Hz, a 5 dB shelf above 4 kHz, firm compression, a short release so "
                "the stabs clip, a light chorus and a little room. Section A has chords and the "
                "left hand only. Section B and the wall add the top note an octave up, 8 dB "
                "under and widened, which is the classic house octave stack. In the last four "
                "bars of the wall the chords gain a fourth voice, where v7 added the fourths "
                "guitar. The space's lone slide becomes one piano grace move into the echo. The "
                "bar-45 stop is now a single piano stab: E3, B3, E4 and G4 over an E2, ringing "
                "into the coda. The coda plays the riff as bare single notes, dry, into the tape "
                "stop, so the piece still ends on its hook. All hook guitars and the organ body "
                "are gone. Their tracks stay in the list, empty, and their notes are still "
                "generated and then dropped, so no timing or noise stream moves. The wall fuzz "
                "stays as the one guitar, 6 dB down, as grit under the piano. The drums, swing, "
                "bass, form, tempo, pads and drone are exactly as in v7. Measured levels: the "
                "piano sits at -1.1 dB in the mid band against the mix in section A, at -1.7 dB "
                "in B and at -1.9 dB in the wall (stabs plus octave), and at -0.2 dB in the coda. "
                "The bass, kick, snare, pads and drone are within 0.1 dB of their v7 levels. The "
                "limiter's peak reduction is 5.4 dB, against 4.5 in v7. The piano carries less "
                "upper-mid energy than the distorted guitars did, so the master gain is 1.4 dB "
                "higher and the untouched drum transients reach the limiter harder. A brighter, "
                "thinner piano EQ was tried, but it raised the limiter and moved the untouched "
                "stems by 0.4 dB, so it was dropped.",
             9: "Variation 2 of 3 of the hook, built on v7 alone. Variation 1 (v8) is left as it "
                "was, and variation 3 will also start from v7. Jonathan rated v7 7/10. He said "
                "the riff was the wrong instrument and lacked texture, and he chose a sampled "
                "house piano. This variation keeps the same rhythm with a new shape. The riff "
                "keeps v7's onsets on steps 2, 4, 8, 9, 11 and 13, all in the bass's gaps, and "
                "its gates. The contour is rewritten for piano as a call in the right hand, high, "
                "answered by a lower chord figure in the left hand. The right hand plays B4, then "
                "the D-to-E move on the snare, then an octave drop on the same pitch: G5 to G4 "
                "in the call bar and A5 to A4 in the answer bar. So v7's G and A peaks are still "
                "there, as the landing notes of the drop. A piano cannot slide, so the slide "
                "becomes a chord move: D5 over A4, then 50 ms later E5 over B4, in parallel "
                "fourths. The first dyad is struck lighter, like a grace note. The left hand "
                "answers on steps 11 and 13, following the pads' harmony. Over E it plays E-A-D "
                "and then E-B in the call bar, so the call still ends on B as a question. In the "
                "answer bar it plays F sharp-A-D and then E-B-E, so the answer resolves to E. "
                "The C and D bars use the same shapes on C and on D. No left-hand note is below "
                "C3, and no chord holds the third of E. Over C the right hand's notes read as "
                "add9 and sixth colours, and over D the G reads as a sus4. The right-hand tops "
                "never change, so the hook recurs identically, and B, the D-to-E landing and the "
                "E answer keep it in the same family. Each stab gets one small timing offset of "
                "2 ms at most. "
                "The piano is voiced as an early-90s house piano. It uses the harder, brighter "
                "sample layers and a very short damper, so the stabs clip. The chain is a "
                "120 Hz high-pass, a 3 dB dip at 300 Hz, a 5 dB shelf above 4 kHz, firm "
                "compression with a 2 ms attack, a shaved top, a light chorus and a little "
                "room. The right hand is stacked an octave below, 6 dB under, throughout. From "
                "section B an octave above is added, 8 dB under and widened. The left hand has "
                "its own track, high-passed at 150 Hz so the bass keeps the low octave, with a "
                "further 4 dB dip at 250 Hz and less room. "
                "By section: A has the right hand with its lower octave and the left hand. The "
                "space has only the chord move into the echo. B adds the upper octave. The wall "
                "has the full stack, and in its last four bars the left hand adds the root an "
                "octave down, where v7 added the fourths guitar. The stop is one ringing chord "
                "of E2, E3, B3, E4 and B4, with no third, rolled 5 ms a note so that its attacks "
                "do not stack on the stop's drums. The coda has the right hand alone, dry and "
                "mono, and then the left hand's E-B-E answer chord held into the tape stop, so "
                "the piece ends on the answer. "
                "All hook guitars and the organ body are gone. As in v8, their notes are still "
                "generated and then dropped, so no timing or noise stream moves. The wall fuzz "
                "stays 6 dB down as the one guitar grit. It keeps only its untouched notes on "
                "steps 2, 4 and 8, which match the new right hand exactly. Its old steps 9, 11 "
                "and 13 would clash with the octave drop and the left hand, so they are removed. "
                "The drums, swing, bass and its lock with the kick, form, tempo, bridge keys, "
                "pads and drone are exactly as in v7. "
                "Measured levels in the mid band against the mix: the right hand is at -1.3 dB "
                "in A, -2.2 dB in B and -2.4 dB in the wall. With its octave stacks the group is "
                "at about -0.7, -1.2 and -1.4 dB. The left hand is at -4.4, -6.8 and -6.8 dB, so "
                "the call leads. The coda piano is at 0.0 dB. The bass is at -3.5 dB, the kick "
                "at -8.3 dB and the drone at -15.7 dB, which are their v7 levels. The first draft "
                "took the limiter to 7.5 dB, because the hardest sample layer's attacks stacked "
                "on the snare and on the stop. A faster compressor attack, a deeper shave, the "
                "rolled stop chord and the lighter grace dyad brought it down to 5.0 dB. That is "
                "better than v8's 5.4 dB but still above v7's 4.5 dB and the 3 dB rule, because "
                "the remaining peaks are the drum and bass transients, which stay untouched. "
                "The low-mid share measures 32.4 per cent, slightly above v8's 31.9 per cent, "
                "because the low end is now limited less.",
             10: "Variation 3 of 3 of the hook, built on v7 alone. Variations 1 and 2 (v8 and v9) "
                 "are left as they were. Jonathan rated v7 7/10 and chose a sampled house piano "
                 "for the hook. For this variation he asked for a new house piano vamp. It is a "
                 "two-bar call and answer of off-beat chord stabs on steps 2, 5, 9, 11 and 13: "
                 "the 'and' of one, the 'e' of two, the 'e' and the 'a' of three, and the 'e' of "
                 "four. None of these meets the bass, the snare or any kick, so the hook still "
                 "lives in the bass's gaps. The stabs are clipped to under half a step, and only "
                 "the answer's last chord is held. A short top line rides on the stabs. The call "
                 "plays B, D, then D moving to E, then D and B, and ends on B as a question. The "
                 "answer plays B, D, E, then a short G (v7's peak note), and resolves on a held "
                 "E. The top line never changes. Only the chords under it move with the E E C D "
                 "cycle, so the hook is recognisable within its first two bars. Under each top "
                 "note sit the two nearest tones of the pad's own chord, a little softer: E "
                 "minor eleventh colours without the third over E, C major seventh and ninth "
                 "over C, and D six-nine and sus shapes over D. No stab note is below B3, and "
                 "the only G over E is the top line's own. A piano cannot slide, so the D-to-E "
                 "slide becomes a chord move on step 9, off the snare: the D chord is struck "
                 "lightly, and 50 ms later the E chord lands at full strength. "
                 "Variations 1 and 2 measured hollow in the upper mids, and in v9 the listening "
                 "model mistook the piano's low octave for the bass. So this piano sits high and "
                 "bright, with no octave below anywhere. It uses the hardest sample layers, a "
                 "very short damper, a 200 Hz high-pass, a 4 dB dip at 300 Hz, a 5 dB bite at "
                 "2.8 kHz and a 5 dB shelf above 4 kHz. Compression is firm and fast, the top "
                 "is shaved, and there is a little chorus and room. The stabs stay centred. "
                 "Section by section: in A, the stabs play with the top line doubled an octave "
                 "up and widened. The space has only the grace dyad, once a bar, into the echo. "
                 "Section B adds the top line two octaves up as sparkle, and a left-hand root "
                 "and fifth at C4 or above on steps 2 and 9. The wall has the full stack, and "
                 "the stabs gain a fourth voice from bar 41, where v7 added the fourths guitar. "
                 "The stop is one ringing E chord with no third (E3, B3, E4, B4 and E5), rolled "
                 "5 ms a note. The coda has the top line alone, dry and mono, with the answer's "
                 "E held into the tape stop, so the piece still ends on its hook. "
                 "All hook guitars and the organ body are gone. As in v8 and v9, their notes "
                 "are still generated and then dropped, so no timing or noise stream moves. The "
                 "wall fuzz's old notes followed v7's contour and would clash with the vamp, so "
                 "they leave too, and the fuzz now plays the new top line an octave up on its "
                 "own seed. It stays at -17 dB, quietly under the piano, about 17 dB below the "
                 "piano in the mid band, so it adds grit without masking the 2.8 kHz bite. The "
                 "drums, swing, bass and its lock with the kick, form, tempo, bridge keys, pads "
                 "and drone are exactly as in v7. "
                 "Measured levels in the mid band against the mix: the stabs are at -2.5 dB in "
                 "A, -3.3 dB in B and -3.4 dB in the wall. With their stacks, the group is at "
                 "about -1.2, -1.7 and -1.9 dB. The octave-up stack is at -7 to -8 dB, the "
                 "sparkle at about -14 dB and the left hand at -11 to -12 dB. The coda piano is "
                 "at -0.1 dB, the echoed grace at -5.2 dB and the stop chord at -6.5 dB. The "
                 "bass is at -3.4 dB, the kick at -8.3 dB and the drone at -15.5 dB, within "
                 "0.2 dB of their v7 levels. The 1-4 kHz share is 14.0 per cent, up from 8.1 in "
                 "v9. The low mids are down from 32.4 to 21.8 per cent, and the spectral "
                 "centroid is up from 611 to 695 Hz. Presence (4-10 kHz) moved only from 2.2 to "
                 "2.5 per cent, short of the 3.5 per cent aimed for, because the piano "
                 "recordings carry little energy there. "
                 "Four drafts set the balance. In the first, the octave stacks measured 7 and "
                 "14 dB under the stabs, so they were raised. The second drew the kick 0.3 dB "
                 "down against the mix, so the piano's weight moved up into the bite and shelf, "
                 "and the stabs came down 1 dB. The limiter's peak reduction is 5.1 dB, against "
                 "5.0 in v9 and 4.5 in v7. Its peaks are the untouched drum and bass "
                 "transients. Softening the stabs beside the kicks was tried in a fourth draft, "
                 "but it raised the master gain and the limiter to 5.2 dB, so it was reverted."}
    if rev >= 6:
        # revision 6 is a new composition with its own builder; revisions 1-5 are unchanged
        return build_v6(rev, notes.get(rev, notes[7]))
    if rev >= 5:
        # revision 5 is a rewrite with its own builder; revisions 1-4 are built above, unchanged
        return build_v5(rev, notes[5])
    return Song("Concrete Tide", "c_concrete_tide", BPM, length, tracks, buses, master, sections,
                SEED, rev, notes.get(rev, ""),
                meta={"bar_starts": [g.bar(i) for i in range(n_bars + 1)]})


def build_v5(rev, note_text):
    """Revision 5 onward: the riff is the hook (R29), the bass bites, the beds are layered (R26)
    and dense 3 is a wall after near-silence (R25). The section map, the contested second, the
    dropped snare, the drifting Euclidean cycles and the composed joins are kept from v4."""
    rng = np.random.default_rng([SEED, 5])          # wood, shaker and fuzz timing
    rng3 = np.random.default_rng([SEED, 3])         # the dense-2 mutation path, as in v3 and v4
    rngp = np.random.default_rng([SEED, 3, 1])      # pipe pitch walk, hat jitter (R17)
    rng_l = np.random.default_rng([SEED, 5, 1])     # left guitar timing
    rng_r = np.random.default_rng([SEED, 5, 2])     # right guitar: its own jitter seed
    rng_c = np.random.default_rng([SEED, 5, 3])     # single and solo guitar timing
    g = Grid(BPM, swing=0.55)
    sec = {"intro": (0, 4), "dense 1": (4, 12), "space 1": (12, 16), "dense 2": (16, 24),
           "space 2": (24, 30), "dense 3": (30, 42), "coda": (42, 50)}
    n_bars, STOP, D2 = 50, 37, nn("D2")
    (bass, grit, kit, hat, snare_fx, wood, pipe, plate, tide, choir, choir_close, lead, fuzz,
     gtr_solo, gtr_c, gtr_l, gtr_r, keys, keys_ghost, air, drone, swell) = ([] for _ in range(22))

    def mode(b):
        """Which second rules a bar: dorian E or phrygian Eb (as v3 and v4)."""
        if sec["dense 2"][0] <= b < sec["dense 2"][1] or 34 <= b < 38 or 40 <= b < 42:
            return "phr"
        return "dor"

    def chord(b):
        return (PROG_PHRYG if mode(b) == "phr" else PROG)[(b // 2) % 4]

    def second(m):
        return nn("Eb4") if m == "phr" else nn("E4")

    def put(dst, b, step, length, pitch, vel, gate, params=None, dt=0.0, jrng=None):
        t = g.t(b, step) + dt
        if jrng is not None:
            t += float(np.clip(jrng.normal(0.0, 0.0012), -0.002, 0.002))   # R11: 2 ms at most
        dst.append(Note(max(0.0, t), g.steps(length) * gate, pitch, vel, dict(params or {})))

    def bass_bar(b, cell, gate=0.6, with_grit=True):
        # clipped notes and a wide accent map; the pluck has no velocity gain, so the accent is
        # also the note's gain
        for s, ln, o, _ in cell:
            a = bass_accent(s)
            put(bass, b, s, ln, D2 + o, a, gate, {"gain": a})
            if with_grit:
                put(grit, b, s, ln, D2 + o, a, gate, {"gain": a})

    SOLO = [(gtr_solo, 0.0, 0.0, rng_c)]
    SINGLE = [(gtr_c, 0.0, 0.0, rng_c)]
    DOUBLE = [(gtr_l, 0.0, 0.0, rng_l), (gtr_r, 0.009, 0.07, rng_r)]   # +9 ms, +7 cents
    KEYS = [(keys, 0.0, 0.0, None)]
    GHOST = [(keys_ghost, 0.0, 0.0, None)]

    def riff(b, m, sect, dsts, octave=0, chop=("D3", "A3")):
        acc = GTR_ACCENTS[sect]
        for s, ln, p in (GTR_A if b % 2 == 0 else GTR_B):
            if p == "chop":
                pitches, gate, params = [nn(x) for x in chop], 0.3, {"mute": 0.7}
            else:
                pitches, gate, params = [second(m) if p == "2nd" else nn(p)], 0.5, {}
            for dst, dt, dsemi, jr in dsts:
                for pp in pitches:
                    put(dst, b, s, ln, pp + octave + dsemi, acc[s], gate, params, dt, jr)

    def fuzz_double(b, m):
        # the fuzz doubles only the riff's tail (steps 10-14), an octave above the guitars
        for s, ln, p in (GTR_A if b % 2 == 0 else GTR_B):
            if 10 <= s <= 14 and p != "chop":
                pitch = (second(m) if p == "2nd" else nn(p)) + 12
                add_cell(fuzz, g, b, [(s, ln, 0, 0.8)], pitch, gate=0.5, jitter=0.0012, rng=rng)

    shaker_pat = euclid(4, 11)
    pipe_pat = euclid(3, 7)
    pipe_prev = [nn("A5")]

    def pipe_global(b, vel=0.7):
        # 3 in 7 on a global sixteenth counter (R16), walking over the section's pitch set
        pset = [nn(x) for x in PIPE_SET[mode(b)]]
        for s in range(16):
            if pipe_pat[(b * 16 + s) % 7]:
                p = walk(pset, pipe_prev[0], 1, rngp)[0]
                pipe_prev[0] = p
                add_cell(pipe, g, b, [(s, 4, 0, vel * (1.0 if s % 4 == 0 else 0.85))], p,
                         params={"decay": 0.8})

    def drums(b, kicks, snare=True, wood_on=True, shaker=True, toms=False, hats=False,
              snare_dst=None):
        if kicks:
            add_hits(kit, g, b, kicks, 36)
        if snare:
            add_hits(kit if snare_dst is None else snare_dst, g, b, [(8, 1.0)], 40)
        if wood_on:
            add_hits(wood, g, b, [(s, 0.6 + 0.3 * (s % 4 == 2)) for s in
                                  euclid_steps(euclid(5, 16, 2))], 76 if b % 2 else 77,
                     jitter=0.002, rng=rng)
        if shaker:
            add_hits(kit, g, b, [(s, 0.35) for s in range(16) if shaker_pat[(b * 16 + s) % 11]],
                     70, jitter=0.002, rng=rng)
        if hats:
            add_hits(hat, g, b, [(s, 0.25 + 0.05 * (s % 4 == 2)) for s in
                                 euclid_steps(euclid(9, 16))], 42, jitter=0.002, rng=rngp)
        if toms:
            for k, s in enumerate(euclid_steps(euclid(3, 16, 3))):
                add_hits(kit, g, b, [(s, 0.7)], [48, 45, 41][k % 3])

    # ---------------------------------------------------------------- intro: the hook, alone
    for b in range(*sec["intro"]):
        if b < 2:
            for s in euclid_steps(euclid(3, 8)):
                add_cell(pipe, g, b, [(s * 2, 4, 0, 0.5 + 0.1 * b)], nn("A5"))
            add_hits(kit, g, b, [(s, 0.25) for s in euclid_steps(euclid(5, 16))], 70)
        else:
            riff(b, "dor", "intro", SOLO)            # stated once, mono and dry
    for p in PROG[0]:
        add_cell(tide, g, 0, [(0, 64, 0, 0.6)], nn(p), gate=1.0)
    add_hits(kit, g, 3, [(12, 0.6)], 49)
    add_cell(tide, g, 3, [(0, 16, 0, 0.7)], nn("D4"), gate=1.0, params={"reverse": True})

    # ---------------------------------------------------------------- dense 1: single guitar
    for b in range(*sec["dense 1"]):
        bass_bar(b, RIFF_1_V3 if b % 2 == 0 else RIFF_2_DOR_V3, with_grit=b >= 6)
        drums(b, KICK_D1_V5, snare=b >= 6, shaker=b >= 8)
        if b >= 10:
            pipe_global(b)
        riff(b, "dor", "dense 1", SINGLE)
        if b % 2 == 0:
            for p in PROG[(b // 2) % 4]:
                add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
    add_cell(plate, g, sec["dense 1"][1] - 1, [(12, 8, 0, 0.55)], nn("D4"))

    # ---------------------------------------------------------------- space 1 (R20)
    for b in range(*sec["space 1"]):
        for s, ln, o, v in SPACE_BASS:
            put(bass, b, s, ln, D2 + o, v, 0.95, {"gain": v})
        add_cell(plate, g, b, [(6, 8, 0, 0.6)], nn("D4") + (b % 2) * 5)
        add_cell(pipe, g, b, [(14, 4, 12, 0.35)], nn("A5"))
        if b >= 14:
            riff(b, "dor", "space", GHOST, octave=12, chop=("A3",))   # the keyboard ghost
    # composed join into dense 2: crash, reversed Eb swell with a 40-cent tape sag (below)
    add_hits(kit, g, sec["dense 2"][0] - 1, [(12, 0.8)], 49)
    add_cell(tide, g, sec["dense 2"][0] - 1, [(0, 16, 0, 0.7)], nn("Eb4"), gate=1.0,
             params={"reverse": True})

    # ---------------------------------------------------------------- dense 2: double-tracked
    ans = list(RIFF_2_PHR_V3)
    for b in range(*sec["dense 2"]):
        if b % 2 == 1 and b > sec["dense 2"][0] + 1:
            ans = mutate_answer(ans, rng3)              # cumulative drift, one change each
        bass_bar(b, RIFF_1_V3 if b % 2 == 0 else ans)
        drums(b, KICK_D2_V5, toms=(b % 4 == 3), hats=True, snare_dst=snare_fx)
        pipe_global(b)
        riff(b, "phr", "dense 2", DOUBLE)
        if b % 2 == 0:
            for p in chord(b):
                add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
    # the lead is reduced to an answer: F, then the E against the phrygian Eb
    add_line(lead, g, sec["dense 2"][0], LEAD_D2_V3[-2:], gate=0.97)
    add_cell(plate, g, sec["dense 2"][1] - 1, [(12, 8, 0, 0.55)], nn("E4"))

    # ---------------------------------------------------------------- space 2: close and dry
    for b in range(*sec["space 2"]):
        if b % 2 == 0:
            for p in ("Bb2", "F3", "C4", "E4"):          # Bb lydian cluster
                add_cell(choir_close, g, b, [(0, 32, 0, 0.55)], nn(p), gate=1.0)
        add_cell(plate, g, b, [(0, 8, 0, 0.45)], nn("E4"))
        if b < sec["space 2"][1] - 1:
            riff(b, "dor", "space", KEYS, octave=12, chop=("A3",))
    b = sec["space 2"][1] - 1
    add_hits(kit, g, b, [(12, 0.9)], 49)
    add_cell(tide, g, b, [(0, 16, 0, 0.8)], nn("D4"), gate=1.0, params={"reverse": True})

    # ---------------------------------------------------------------- dense 3: build, hole, wall
    for b in range(*sec["dense 3"]):
        m = mode(b)
        if b == STOP:
            # near-silence: the plate, the reversed breath swell (below) and one dry chop
            add_cell(plate, g, b, [(0, 8, 0, 0.6)], nn("D4"))
            put(gtr_solo, b, 14, 2, second(m), 0.95, 0.3, {"mute": 0.6}, 0.0, rng_c)
            continue
        if b % 2 == 0:
            cell = RIFF_1_V3
        elif b >= STOP + 1:
            cell = RIFF_CLIMB
        else:
            cell = RIFF_2_DOR_V3 if m == "dor" else RIFF_2_PHR_V3
        if b == STOP - 1:
            bass_bar(b, cell, with_grit=False)          # the clean bass and the riff, exposed
            riff(b, m, "dense 3", SOLO)
            continue
        bass_bar(b, cell)
        if b >= STOP + 1:
            drums(b, KICK_D3, shaker=False, toms=True, hats=True)
            add_hits(kit, g, b, [(0, 0.75)], 49)        # noise burst instead of a crash (R28)
        else:
            drums(b, KICK_D3, toms=(b % 2 == 1), hats=True)
        if b >= 34:
            pipe_global(b)
            fuzz_double(b, m)
        riff(b, m, "dense 3", DOUBLE)
        if b in (30, 32, 34, 40):
            for p in chord(b):
                add_cell(choir, g, b, [(0, 32, 0, 0.5)], nn(p) + 12, gate=1.0)
    # the climax chord is laid under the stop, so it is fully open on the bar-38 downbeat
    for p in chord(STOP + 1):
        add_cell(choir, g, STOP, [(0, 48, 0, 0.55)], nn(p) + 12, gate=1.0)

    # ---------------------------------------------------------------- coda: subtraction
    for i, b in enumerate(range(*sec["coda"])):
        cell = RIFF_1_V3 if b % 2 == 0 else RIFF_2_DOR_V3
        if b >= 46:                                     # the minor third leaves before the F#
            cell = [c for c in cell if c[2] != 3]
        keep = max(2, len(cell) - i)                    # one onset fewer per bar (R3)
        bass_bar(b, cell[:keep], gate=0.6 if i < 2 else 0.8, with_grit=i < 2)
        if i < 2:
            riff(b, "dor", "coda", SINGLE)              # the piece ends on its hook
        if i < 4:
            for s in euclid_steps(euclid(3, 8)):
                add_cell(pipe, g, b, [(s * 2, 4, 0, 0.5 - 0.08 * i)], nn("A5"))
        if i % 2 == 0:
            for p in (CODA_MAJOR if b >= 46 else PROG[0]):
                add_cell(tide, g, b, [(0, 32, 0, 0.5)], nn(p), gate=1.0)
    add_cell(bass, g, n_bars - 1, [(8, 8, 12, 0.6)], D2, gate=1.0, params={"gain": 0.8})
    add_cell(plate, g, n_bars - 1, [(8, 8, 0, 0.5)], nn("D4"))

    # ---------------------------------------------------------------- the bed (R26)
    def span(dst, b0, b1, pitch, vel, params=None):
        dst.append(Note(g.bar(b0), g.bar(b1) - g.bar(b0), pitch, vel, dict(params or {})))

    # breath noise round 3.5 kHz, every section, at a level per section; the wall's note starts
    # under the stop and opens with it
    for b0, b1, v in [(0, 4, 0.45), (4, 12, 0.75), (12, 16, 0.55), (16, 24, 0.8), (24, 30, 0.5),
                      (30, 36, 0.85), (37, 42, 1.0), (42, 50, 0.5)]:
        span(air, b0, b1, 105, v)
    # D and A drone: dense sections and space 1 only; out of space 2, leaves early in the coda
    for b0, b1, v in [(4, 12, 0.9), (12, 16, 0.8), (16, 24, 1.0), (30, 36, 1.0), (36, 42, 1.15),
                      (42, 46, 0.7)]:
        for p in ("D2", "A2"):
            span(drone, b0, b1, nn(p), 1.0, {"gain": v})

    def copies(src, b0, b1):
        return [replace(n, params=dict(n.params)) for n in src
                if g.bar(b0) - 0.05 <= n.t < g.bar(b1) - 0.05 and not n.params.get("reverse")]

    # the crushed layer copies the sounding pad chords of the dense sections (taken before the
    # tide's tape-sag offsets, as the copy is not varispeeded) and the coda's first chord
    crush = (copies(tide, 4, 12) + copies(tide, 16, 24) + copies(choir, 30, 42)
             + copies(tide, 42, 44))
    # bar 37's reversed breath swell into the wall (its level ramp is the track automation)
    swell.append(Note(g.bar(STOP), g.bar(STOP + 1) - g.bar(STOP), nn("D6"), 0.9,
                      {"reverse": True}))
    # the wall's crushed kit-bus copy, mixed in parallel at about 30 per cent
    t38, t42 = g.bar(STOP + 1), g.bar(42)
    kit_crush = [replace(n, params=dict(n.params)) for n in kit if t38 - 0.01 <= n.t < t42]

    # ---------------------------------------------------------------- tape sags (R14), as v4
    t29, t30 = g.bar(29), g.bar(30)
    vs = [(t29, 0.0), (t29 + g.steps(12), -70.0), (t30, 0.0)]
    t15, t16 = g.bar(15), g.bar(16)
    vs15 = [(t15, 0.0), (t15 + g.steps(12), -40.0), (t16, 0.0)]

    def sag_lag(pts, a, b):
        tt = np.linspace(a, b, 4001)
        return float(np.mean(1 - 2 ** (np.interp(tt, [p[0] for p in pts], [p[1] for p in pts])
                                       / 1200)) * (b - a))

    lag29, lag15 = sag_lag(vs, t29, t30), sag_lag(vs15, t15, t16)
    for n in tide:                       # the slowed read head lands late: start early
        if n.t >= t29 - 0.01:
            n.t -= lag29
    for n in tide:
        if n.t >= t15 - 0.01:
            n.t -= lag15

    # ---------------------------------------------------------------- mix
    length = g.bar(n_bars)
    end = length + 5.0
    t36 = g.bar(STOP - 1)
    # the hole before the wall: sustained layers are cut at bar 36 and reopen on bar 38 (R25)
    hole = [(0.0, 0.0), (t36, 0.0), (t36 + 0.25, -50.0), (t38 - 0.03, -50.0), (t38, 0.0)]
    duck = ("kick", 3.0, 0.18)           # the sustained layers breathe with the kick (R28)
    kit_patch = {**PATCHES["kit"],
                 "snare_drop": {**PATCHES["kit"]["snare_drop"], "noise_f": 4500, "drop": 16,
                                "decay": 0.55},
                 "hat": {"decay": 0.04, "tune": 1.6, "bp": 9500}}
    pad_cut = ["peak", 2500, 0.9, -2.0]
    amp = ("c_amp", PATCHES_V5["amp"])
    gtr_eq = ("eq", {"bands": [["peak", 220, 0.8, -3.0]]})   # leave the low octave to the bass
    gtr_sends = {"room": 0.15, "echo": 0.1}
    tracks = [
        Track("bass", "pluck", PATCHES["bass"], bass, gain_db=0.0, program=35,
              stereo={"chorus": {"depth_ms": 1.5, "rate": 0.5, "mix": 0.25}},
              inserts=[("comp", {"thresh_db": -14, "ratio": 3, "attack": 0.01, "release": 0.12}),
                       ("eq", {"bands": [["peak", 800, 1.0, 1.0], ["lowshelf", 100, 0.7, 0.5]]}),
                       ("softclip", {"shave_db": 3.0})],
              duck=("kick", 4.0, 0.08), sends={"room": 0.1}),
        # the grit: same notes, fuzzed and band-limited above the sub (R24, M1)
        Track("bass_grit", "pluck", PATCHES_V5["grit"], grit, gain_db=-10.0, program=33,
              inserts=[("fuzz", {"gain": 8, "tone": 2500, "out": 0.5}), ("hp", {"f": 280.0}),
                       ("lp", {"f": 3200.0}), ("eq", {"bands": [["peak", 1800, 1.0, -2.5]]})],
              duck=("kick", 4.0, 0.08)),
        Track("kick", "drums", kit_patch, [n for n in kit if n.pitch == 36], gain_db=-5.0,
              inserts=[("softclip", {"shave_db": 3.0})]),
        Track("kit", "drums", kit_patch, [n for n in kit if n.pitch != 36], gain_db=-5.0,
              pan=0.1, sends={"plate": 0.35}, inserts=[("softclip", {"shave_db": 3.0})]),
        # the wall's kit, crushed in parallel (R28)
        Track("kit_crush", "drums", kit_patch, kit_crush, gain_db=-15.5, stereo={"haas": 11},
              inserts=[("decimate", {"hold": 3, "bits": 10}), ("softclip", {"shave_db": 4.0}),
                       ("hp", {"f": 150.0})]),
        Track("hat", "drums", kit_patch, hat, gain_db=-1.0, pan=-0.2, sends={}),
        Track("snare_fx", "drums", kit_patch, snare_fx, gain_db=-5.0, pan=0.1,
              sends={"plate": 0.35}, inserts=[("decimate", {"hold": 6, "bits": 8})]),
        Track("wood", "drums", PATCHES["wood"], wood, gain_db=-11.5, pan=-0.45,
              sends={"room": 0.3}),
        Track("pipe", "metal", {**PATCHES["pipe"], "click": 0.15}, pipe, gain_db=-12.0, pan=0.5,
              sends={"plate": 0.3, "echo": 0.2}, program=14,
              inserts=[("ringmod", {"freq": 311.13, "mix_points": [
                  (0.0, 0.0), (t38 - 0.02, 0.0), (t38, 0.6), (t42, 0.6), (t42 + 0.4, 0.0)]})]),
        Track("plate", "metal", PATCHES["plate"], plate, gain_db=-11.0, pan=-0.3,
              sends={"plate": 0.5, "echo": 0.3}, program=112),
        Track("tide", "pad", PATCHES_V5["tide"], tide, gain_db=-12.0,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.11, "mix": 0.7}},
              sends={"hall": 0.45}, program=89, duck=duck,
              inserts=[("eq", {"bands": [["peak", 350, 1.0, -3.0], pad_cut]}),
                       ("varispeed", {"points": vs15 + vs}),
                       ("tape", {"drive": 0.9, "wow": 0.35, "flutter": 0.1, "hf": 8000.0,
                                 "bump_db": 0.0})]),
        Track("choir", "pad", PATCHES["choir"], choir, gain_db=-13.0,
              stereo={"chorus": {"depth_ms": 6.0, "rate": 0.12, "mix": 0.7}},
              sends={"hall": 0.5}, program=52, duck=duck, auto=hole,
              inserts=[("eq", {"bands": [["peak", 350, 1.0, -3.0], pad_cut]})]),
        # space 2's choir: close and dry, no hall (R25)
        Track("choir_close", "pad", PATCHES["choir"], choir_close, gain_db=-12.0,
              stereo={"chorus": {"depth_ms": 3.0, "rate": 0.09, "mix": 0.5}}, program=52,
              inserts=[("eq", {"bands": [["peak", 350, 1.0, -3.0], pad_cut]})]),
        Track("lead", "lead", PATCHES["lead"], lead, gain_db=-11.0, pan=0.2,
              sends={"echo": 0.4, "hall": 0.25}, program=81,
              inserts=[("eq", {"bands": [["peak", 400, 1.0, -3.0]]}),
                       ("tape", {"drive": 0.9, "wow": 0.25, "flutter": 0.1, "hf": 9000.0,
                                 "bump_db": 0.0})]),
        Track("fuzz", "pluck", PATCHES["fuzz"], fuzz, gain_db=-17.0, stereo={"haas": 14},
              inserts=[("fuzz", {"gain": 6, "tone": 2800, "out": 0.6}),
                       ("eq", {"bands": [["peak", 2800, 1.2, -4.0], ["hp", 220, 0.7],
                                         ["lp", 4500, 0.7]]})],
              sends={"plate": 0.25, "echo": 0.15}, program=30),
        # the riff (R29): solo statement, single guitar, and the double-tracked pair
        Track("gtr_solo", "c_guitar", PATCHES_V5["guitar"], gtr_solo, gain_db=-10.0,
              inserts=[amp, gtr_eq], program=29),
        Track("gtr_c", "c_guitar", PATCHES_V5["guitar"], gtr_c, gain_db=-10.5, pan=-0.3,
              inserts=[amp, gtr_eq], sends=gtr_sends, program=29),
        Track("gtr_l", "c_guitar", PATCHES_V5["guitar"], gtr_l, gain_db=-12.5, pan=-0.7,
              inserts=[amp, gtr_eq], sends=gtr_sends, program=29),
        Track("gtr_r", "c_guitar", PATCHES_V5["guitar"], gtr_r, gain_db=-12.5, pan=0.7,
              inserts=[amp, gtr_eq], sends=gtr_sends, program=29),
        # the riff as a degraded keyboard (R27): dry in space 2, an echoed ghost in space 1
        Track("keys", "c_keys", PATCHES_V5["keys"], keys, gain_db=-9.0, pan=0.1, program=4,
              inserts=[("tape", {"drive": 0.8, "wow": 0.5, "flutter": 0.15, "hf": 9000.0,
                                 "bump_db": 0.0})]),
        Track("keys_ghost", "c_keys", PATCHES_V5["keys"], keys_ghost, gain_db=-16.0, pan=-0.2,
              sends={"echo": 0.35, "hall": 0.2}, program=4,
              inserts=[("tape", {"drive": 0.8, "wow": 0.5, "flutter": 0.15, "hf": 9000.0,
                                 "bump_db": 0.0})]),
        # the bed (R26): three filtered layers on incommensurate sweeps (R16)
        Track("bed_air", "breath", PATCHES_V5["air"], air, gain_db=-26.0, program=122,
              stereo={"chorus": {"depth_ms": 4.0, "rate": 0.07, "mix": 0.8}},
              sends={"hall": 0.15}, duck=duck, auto=hole,
              inserts=[("hp", {"f": 2000.0}), ("lp", {"f": 6000.0}),
                       ("lp_auto", {"points": lfo_points(13.7, 2600.0, 6500.0, end), "q": 0.7})]),
        Track("bed_drone", "pad", PATCHES_V5["drone"], drone, gain_db=-21.0, program=89,
              stereo={"chorus": {"depth_ms": 7.0, "rate": 0.09, "mix": 0.8}},
              sends={"hall": 0.25}, duck=duck, auto=hole,
              inserts=[("tape", {"drive": 1.2, "wow": 0.5, "flutter": 0.05, "hf": 6000.0,
                                 "bump_db": 0.0}),
                       ("hp", {"f": 160.0}),
                       ("lp_auto", {"points": lfo_points(21.1, 500.0, 1800.0, end, 1.3),
                                    "q": 0.8})]),
        Track("bed_crush", "pad", PATCHES_V5["tide"], crush, gain_db=-20.0, program=95,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.13, "mix": 0.8}},
              duck=duck, auto=hole,
              inserts=[("decimate", {"hold": 8, "bits": 6}), ("hp", {"f": 1500.0}),
                       ("lp_auto", {"points": lfo_points(34.3, 2500.0, 7500.0, end, 2.1),
                                    "q": 0.7})]),
        Track("swell", "breath", PATCHES_V5["swell"], swell, gain_db=-8.0, program=122,
              sends={"hall": 0.3},
              auto=[(0.0, -30.0), (g.bar(STOP), -30.0), (t38 - 0.005, 0.0), (t38 + 0.01, -60.0)]),
    ]
    buses = {
        "room": {"type": "reverb", "ir": {"decay": 0.8, "predelay": 0.008, "tone": 7000,
                                          "size": 0.6}, "return_db": -8},
        "plate": {"type": "reverb", "ir": {"decay": 2.4, "predelay": 0.02, "tone": 7500},
                  "return_db": -5},
        "hall": {"type": "reverb", "ir": {"decay": 5.0, "predelay": 0.035, "tone": 5500,
                                          "size": 1.8}, "return_db": -4},
        "echo": {"type": "echo", "echo": {"time_s": g.steps(6), "feedback": 0.5, "repeats": 6,
                                          "tone": 3200, "spread": 0.7, "wow": 1.2, "sat": 0.4},
                 "return_db": -6},
    }
    master = {"target_lufs": -14.0, "ceiling": -1.5, "tail": 5.0, "linear_phase_mono": True,
              "eq": [["hp", 28, 0.7], ["highshelf", 7000, 0.7, 2.0]],
              "comp": {"thresh_db": -18, "ratio": 1.5, "attack": 0.03, "release": 0.25},
              "tape": {"drive": 0.7, "wow": 0.08, "flutter": 0.08, "hiss_db": -78,
                       "hf": 14500, "bump_db": 1.2},
              "true_peak": True}
    sections = [(name, g.bar(a), g.bar(b)) for name, (a, b) in sec.items()]
    return Song("Concrete Tide", "c_concrete_tide", BPM, length, tracks, buses, master, sections,
                SEED, rev, note_text,
                meta={"bar_starts": [g.bar(i) for i in range(n_bars + 1)]})


def build_v6(rev, note_text):
    """Revision 6 onward: a new composition in E at 88 BPM. A call-and-response guitar hook in
    the gaps of a sub-heavy sixteenth bass, a real backbeat with a tritone clang, and an arc of
    groove, space, wider groove, empty bridge, wall, hard stop and tape stop (bars are 0-based
    here: index b is bar b + 1 of the score)."""
    g = Grid(BPM, swing=0.54)
    n_bars = 48
    rng_c = np.random.default_rng([SEED, 6, 1])     # single, echo and DI guitar timing
    rng_l = np.random.default_rng([SEED, 6, 2])     # left guitar
    rng_r = np.random.default_rng([SEED, 6, 3])     # right guitar: its own jitter seed (R11)
    rng_x = np.random.default_rng([SEED, 6, 4])     # fuzz and fourths guitar
    rng_h = np.random.default_rng([SEED, 6, 5])     # hats
    sec = {"intro": (0, 2), "groove": (2, 4), "A": (4, 12), "space": (12, 16), "B": (16, 24),
           "bridge": (24, 30), "build": (30, 32), "wall": (32, 44), "stop": (44, 45),
           "coda": (45, 47), "tape stop": (47, 48)}
    names = ["bass", "grit", "kick", "snare", "snare_noise", "snare_sheet", "clap", "hat", "toms",
             "heart", "anvil", "tick", "noise_hit", "rev_snare", "gtr_c", "gtr_echo", "gtr_l",
             "gtr_r", "gtr_4th", "fuzz", "gtr_fb", "gtr_fb2", "gtr_rev", "gtr_di", "keys", "pad",
             "pad_wide", "pad_bridge", "strings", "drone", "bed"]
    T = {k: [] for k in names}
    v7 = rev >= 7
    if v7:
        # the hook's new layers, each on its own timing seed so no v6 seed is disturbed
        T.update({k: [] for k in V7_NAMES})
        rng_cu = np.random.default_rng([SEED, 6, 6])    # section A's unison partner
        rng_lu = np.random.default_rng([SEED, 6, 7])    # left unison partner
        rng_ru = np.random.default_rng([SEED, 6, 8])    # right unison partner
        rng_o = np.random.default_rng([SEED, 6, 9])     # octave layer

    def put(name, b, step, length, pitch, vel, gate=1.0, params=None, dt=0.0, jr=None):
        t = g.t(b, step) + dt
        if jr is not None:
            t += float(np.clip(jr.normal(0.0, 0.0012), -0.002, 0.002))   # R11: 2 ms at most
        T[name].append(Note(max(0.0, t), g.steps(length) * gate, pitch, vel, dict(params or {})))

    def span(name, b0, s0, b1, s1, pitch, vel=1.0, params=None):
        t0, t1 = g.t(b0, s0), g.t(b1, s1)
        T[name].append(Note(t0, t1 - t0, pitch, vel, dict(params or {})))

    def harmony(b):
        if 16 <= b < 24:
            return CYCLE6[(b - 16) % 4]
        if 32 <= b < 44:
            return CYCLE6[(b - 32) % 4]
        return "E"

    def ghost_on(b):
        return (b + 1) % 4 != 0                    # every fourth bar drops the step-3 ghost (R3)

    def bass_bar(b, dive=False, grit=False, upto=16, extra=None):
        ch = harmony(b)
        root = BASS_ROOT6[ch]
        for s, off, ln, v in (BASS_E6 if ch == "E" else BASS_CD6):
            if s >= upto or (dive and s >= 12) or (s == 3 and not ghost_on(b)):
                continue
            pitch = root + off
            pr = {"sub_oct": 0 if (ch == "E" or pitch < 36) else -1}
            if extra:
                pr.update(extra(b, s))
            gate = 0.9 if s in (0, 10) else 0.65
            put("bass", b, s, ln, pitch, v, gate, pr)
            if grit:
                put("grit", b, s, ln, pitch, v, gate, pr)
        if dive:
            # steps 12-15 become one E2-to-E1 dive into the next downbeat
            pr = {"dive_to": 28, "dive_s": 0.3}
            put("bass", b, 12, 4, 40, 0.9, 1.0, pr)
            if grit:
                put("grit", b, 12, 4, 40, 0.9, 1.0, pr)

    def hook(name, b, kind=None, dt=0.0, dc=0.0, jr=None, octave=0, vel=1.0, fourths=False,
             skip=()):
        kind = kind or ("call" if b % 2 == 0 else "answer")
        for s, p, ln, v, to in (CALL6 if kind == "call" else ANSWER6):
            if s in skip:
                continue
            if fourths:
                p, to = FOURTH_BELOW6[p], (FOURTH_BELOW6[to] if to is not None else None)
            pr = {} if to is None else {"slide_to": to + octave + dc}
            gate = 0.9 if (to is not None or ln >= 2) else 0.55
            put(name, b, s, ln, p + octave + dc, v * vel, gate, pr, dt, jr)

    def kicks(b, pairs):
        for s, v in pairs:
            put("kick", b, s, 1, 36, v, 0.5)

    def backbeat(b, steps=(4, 12), clap=False):
        for s in steps:
            put("snare", b, s, 1, 40, 1.0, 0.5)
            put("snare_noise", b, s, 1, 38, 0.9, 0.5)
            put("snare_sheet", b, s, 2, nn("Bb3"), 1.0, 1.0)     # the tritone clang
            if clap:
                put("clap", b, s, 1, 39, 0.9, 0.5)

    def hat_bar(b, upto=16):
        for s in range(upto):
            put("hat", b, s, 1, 42, 0.5 if s % 2 == 0 else 0.22, 0.5, None, 0.0, rng_h)

    def tom_fill(b, order):
        for s, gm in zip((8, 10, 12, 13, 14, 15), order):
            put("toms", b, s, 1, gm, 0.8, 0.5)

    def chords(name, b0, b1, cycle=None, gain=1.0):
        """Pad chords from bar b0 to b1: one chord per harmony run."""
        b = b0
        while b < b1:
            ch = harmony(b) if cycle is None else cycle
            e = b + 1
            while e < b1 and (harmony(e) if cycle is None else cycle) == ch:
                e += 1
            vc = PAD_VOICING6[ch]
            for p in vc:
                span(name, b, 0, e, 0, nn(p), 1.0, {"gain": gain * (1.0 if len(vc) == 5 else 1.1)})
            b = e

    # ---------------------------------------------------------------- bars 1-2: intro
    span("gtr_fb", 0, 0, 2, 2, nn("E5"), 0.9, {"feedback": True, "swell": 4.0})
    span("bass", 0, 0, 2, 0, 28, 0.55, {"a": 4.5, "sub_only": True})    # soft E1 sub swell
    for b0, s0, b1, v in [(0, 0, 4, 0.2), (4, 0, 12, 0.5), (16, 0, 24, 1.0), (31, 8, 44, 1.0)]:
        span("bed", b0, s0, b1, 0, 107, v)          # static bed: -6 dB in A, full in B and wall

    # ---------------------------------------------------------------- bars 3-4: the groove
    for b in range(2, 4):
        bass_bar(b, dive=(b == 3))
        kicks(b, K_A6 + ([(7, 0.6)] if b % 2 else []))
        hat_bar(b)
    span("gtr_rev", 3, 0, 4, 0, 59, 0.9, {"reverse": True, "t60": 3.5})

    # ---------------------------------------------------------------- bars 5-12: section A
    for b in range(4, 12):
        bass_bar(b, dive=(b == 11))
        kicks(b, K_A6 + ([(7, 0.6)] if b % 2 else []))
        backbeat(b)
        hat_bar(b)
        hook("gtr_c", b, jr=rng_c)
        if v7:
            hook("gtr_c_uni", b, dc=-0.11, dt=0.006, jr=rng_cu)    # 11 cents flat, 6 ms late
            hook("gtr_oct", b, octave=12, jr=rng_o)
            hook("hook_body", b)
    chords("pad", 4, 12)
    for p in ("E2", "B2"):
        span("drone", 4, 0, 12, 0, nn(p))

    # ---------------------------------------------------------------- bars 13-16: space
    for i, b in enumerate(range(12, 16)):
        c0, c1 = 160.0 * 5.0 ** (i / 4), 160.0 * 5.0 ** ((i + 1) / 4)   # 160 Hz opening to 800
        if b == 15:
            put("bass", b, 0, 12, 28, 0.9, 1.0, {"sweep": (c0, c1), "fenv": 150.0})
            put("bass", b, 12, 4, 40, 0.9, 1.0, {"dive_to": 28, "dive_s": 0.3})
        else:
            put("bass", b, 0, 16, 28, 0.9, 0.97, {"sweep": (c0, c1), "fenv": 150.0})
        for s in (0, 3):
            put("heart", b, s, 1, 41, 0.5, 0.5)
        put("gtr_echo", b, 4, 2, 62, 1.0, 0.9, {"slide_to": 64}, 0.0, rng_c)
        if v7:
            put("hook_body", b, 4, 2, 62, 0.45, 0.9, {"slide_to": 64})
    span("pad", 12, 0, 16, 0, nn("B3"), 1.0, {"gain": 1.4})
    t_rs = g.t(15, 8)
    T["rev_snare"].append(Note(t_rs, g.bar(16) - t_rs, 40, 0.8, {"reverse": True}))
    span("gtr_rev", 15, 0, 16, 0, 59, 0.95, {"reverse": True, "t60": 3.5})

    # ---------------------------------------------------------------- bars 17-24: section B
    for b in range(16, 24):
        bass_bar(b, dive=(b == 23), grit=True)
        kicks(b, K_B6)
        backbeat(b)
        hat_bar(b)
        put("anvil", b, 6, 1, 108, 0.7, 0.5)
        if b in (19, 23):
            put("snare", b, 15, 1, 40, 0.3, 0.5)
            put("snare_noise", b, 15, 1, 38, 0.3, 0.5)
        if b == 23:
            tom_fill(b, (48, 48, 45, 45, 41, 41))
        hook("gtr_l", b, jr=rng_l)
        hook("gtr_r", b, dt=0.009, dc=0.07, jr=rng_r)              # +9 ms, +7 cents
        if v7:
            hook("gtr_l_uni", b, dc=-0.09, dt=0.005, jr=rng_lu)
            hook("gtr_r_uni", b, dc=0.16, dt=0.005, jr=rng_ru)
            hook("gtr_oct", b, octave=12, jr=rng_o)
            hook("hook_body", b)
    chords("pad_wide", 16, 24)
    for p in ("E2", "B2"):
        span("drone", 16, 0, 24, 0, nn(p))

    # ---------------------------------------------------------------- bars 25-30: bridge
    # the keys play the hook in augmentation: each sixteenth becomes an eighth
    for kind, b0 in (("call", 24), ("answer", 26)):
        for s, p, ln, v, to in (CALL6 if kind == "call" else ANSWER6):
            t = g.bar(b0) + 2 * s * g.step_s
            dur = 2 * ln * g.step_s * (0.9 if (to is not None or ln >= 2) else 0.6)
            T["keys"].append(Note(t, dur, p + 12, v, {} if to is None else {"slide_to": to + 12}))
    for p in CMAJ9_6:
        span("pad_bridge", 24, 0, 26, 0, nn(p), 1.0, {"gain": 1.0})
    for p in EM11_6:
        span("pad_bridge", 26, 0, 31, 8, nn(p), 1.0, {"gain": 1.0})

    def opening(b, s):
        frac = ((b - 28) * 16 + s) / 32.0
        return {"cut": 200.0 * 6.0 ** frac, "fenv": 250.0}       # low-pass 200 Hz to 1.2 kHz

    for b in (28, 29):
        bass_bar(b, dive=(b == 29), extra=opening)
        kicks(b, [(0, 1.0)])
        for s in range(16):
            lvl = 0.1 + 0.3 * ((b - 28) * 16 + s) / 31.0          # hats fade in, .1 to .4
            put("hat", b, s, 1, 42, lvl if s % 2 == 0 else lvl * 0.44, 0.5, None, 0.0, rng_h)

    # ---------------------------------------------------------------- bars 31-32: build
    bass_bar(30)
    kicks(30, [(0, 1.0), (4, 0.85), (8, 0.9), (12, 0.85)])
    backbeat(30, steps=(4,))
    hat_bar(30)
    for k in range(16):                              # sixteenth snare roll, .3 rising to .9
        b, s = (30, 8 + k) if k < 8 else (31, k - 8)
        v = 0.3 + 0.6 * k / 15
        put("snare", b, s, 1, 40, v, 0.5)
        put("snare_noise", b, s, 1, 38, v, 0.5)
    kicks(31, [(0, 1.0), (4, 0.9)])
    bass_bar(31, upto=8)
    hat_bar(31, upto=8)
    span("gtr_rev", 31, 0, 32, 0, 59, 1.0, {"reverse": True, "t60": 3.5})   # R19

    # ---------------------------------------------------------------- bars 33-44: the wall
    for b in range(32, 44):
        bass_bar(b, dive=(b == 43), grit=True)
        kicks(b, K_C6)
        backbeat(b, clap=True)
        hat_bar(b)
        for s in range(16):
            if (b * 16 + s) % 3 == 0:                # 3 against 4 on a global counter (R8)
                put("tick", b, s, 1, 112, 0.4, 0.5)
        if b in (32, 36, 40):
            put("noise_hit", b, 0, 1, 49, 0.9, 0.5)  # noise burst in place of a crash (R28)
        if b == 39:
            tom_fill(b, (41, 41, 45, 45, 48, 48))
        if b == 43:
            tom_fill(b, (48, 48, 45, 45, 41, 41))
        hook("gtr_l", b, jr=rng_l)
        hook("gtr_r", b, dt=0.009, dc=0.07, jr=rng_r)
        hook("fuzz", b, octave=12, dt=0.004, jr=rng_x)
        if b >= 40:
            hook("gtr_4th", b, fourths=True, dt=0.003, jr=rng_x)
        if v7:
            # the wall's octave layer is the existing fuzz, so only unison and body are added
            hook("gtr_l_uni", b, dc=-0.09, dt=0.005, jr=rng_lu)
            hook("gtr_r_uni", b, dc=0.16, dt=0.005, jr=rng_ru)
            hook("hook_body", b)
    # the wall's sustained layers start under the bar-32 hole, so they are open on the downbeat
    pads = [(31, 8, 34, "E"), (34, 0, 35, "C"), (35, 0, 36, "D"), (36, 0, 38, "E"),
            (38, 0, 39, "C"), (39, 0, 40, "D"), (40, 0, 42, "E"), (42, 0, 43, "C"),
            (43, 0, 44, "D")]
    for b0, s0, b1, ch in pads:
        vc = PAD_VOICING6[ch]
        for p in vc:
            span("pad_wide", b0, s0, b1, 0, nn(p), 1.0, {"gain": 1.0 if len(vc) == 5 else 1.1})
    for b0, s0, b1, p in [(31, 8, 36, "B4"), (36, 0, 38, "C5"), (38, 0, 40, "D5"),
                          (40, 0, 44, "E5")]:
        span("strings", b0, s0, b1, 0, nn(p))
    for p in ("E2", "B2"):
        span("drone", 31, 8, 44, 0, nn(p))
    span("gtr_fb2", 40, 0, 44, 0, nn("B4"), 0.9, {"feedback": True, "swell": 2.5})
    t33, t45 = g.bar(32), g.bar(44)
    crush = [replace(n, params=dict(n.params)) for k in ("kick", "snare", "clap")
             for n in T[k] if t33 - 0.01 <= n.t < t45]

    # ---------------------------------------------------------------- bar 45: hard stop
    # every voice lands together, so each hit is a little softer than in the wall
    kicks(44, [(0, 0.9)])
    put("noise_hit", 44, 0, 1, 49, 0.8, 0.5)
    put("snare", 44, 0, 1, 40, 0.8, 0.5)
    put("snare_sheet", 44, 0, 2, nn("Bb3"), 1.0, 1.0)          # the clang rings out
    put("bass", 44, 0, 16, 28, 0.8, 1.0, {"sub_oct": 0})
    # E power chord, left and right; three strings summed peak high in the clean DI copy, so
    # the chord is picked softer (the amp still saturates it fully)
    for p in (52, 59, 64):
        put("gtr_l", 44, 0, 16, p, 0.6, 0.95, {"t60": 3.0}, 0.0, rng_l)
        put("gtr_r", 44, 0, 16, p + 0.07, 0.6, 0.95, {"t60": 3.0}, 0.009, rng_r)
    for p in (64, 71):
        put("fuzz", 44, 0, 16, p, 0.7, 0.95, {"t60": 3.0}, 0.004, rng_x)

    # ---------------------------------------------------------------- bars 46-48: coda
    hook("gtr_di", 45, "call", jr=rng_c)
    hook("gtr_di", 46, "answer", jr=rng_c, skip=(13,))
    t_last = g.t(46, 13)
    T["gtr_di"].append(Note(t_last, g.bar(48) - t_last, 64, 0.9, {"t60": 4.0}))
    put("bass", 47, 0, 16, 28, 0.7, 1.0, {"sub_oct": 0})

    # ---------------------------------------------------------------- rev 8: house piano
    v8 = rev == 8
    if v8:
        # the guitar hook was generated above exactly as in v7, so every shared timing seed
        # (the fuzz's above all) draws as before; its notes are then dropped here
        for k in V8_DROPPED:
            T[k] = []
        T.update({k: [] for k in V8_NAMES})
        rng_pn = np.random.default_rng([SEED, 8, 1])    # one timing offset per stab (R11)

        def pnote(name, t, dur, pitch, vel, params=None):
            T[name].append(Note(max(0.0, t), dur, pitch, vel, dict(params or {})))

        def hook_piano(b, kind=None, n_inner=2, octave=None, chords=True, name="piano",
                       skip=()):
            """hook()'s onsets, lengths, gates and velocities as chord stabs: the riff note on
            top, close chord tones from the pad voicing below at 0.8x, the root in the left
            hand on the quarter-beat steps at 0.7x. The slide becomes a 50 ms grace note."""
            kind = kind or ("call" if b % 2 == 0 else "answer")
            ch = harmony(b)
            pcs = {nn(x) % 12 for x in PAD_VOICING6[ch]}
            for s, p, ln, v, to in (CALL6 if kind == "call" else ANSWER6):
                if s in skip:
                    continue
                gate = 0.9 if (to is not None or ln >= 2) else 0.55
                t = g.t(b, s) + float(np.clip(rng_pn.normal(0.0, 0.0012), -0.002, 0.002))
                dur = g.steps(ln) * gate
                tops = [(p, t, dur)] if to is None else [(p, t, GRACE_S),
                                                         (to, t + GRACE_S, dur - GRACE_S)]
                for tp, tt, dd in tops:
                    pnote(name, tt, dd, tp, v)
                    if octave:
                        pnote(octave, tt, dd, tp + 12, v)
                if not chords:
                    continue
                inner = stab_voicing(p, pcs, n_inner)
                for c in inner:
                    pnote(name, t, dur, c, 0.8 * v)
                if s % 4 == 0:
                    pnote(name, t, dur, min(LH_ROOT8[ch], min(inner) - 1), 0.7 * v)

        # one track per section, so each section's piano level is measured on its own
        for b in range(4, 12):                           # section A: chords and left hand
            hook_piano(b)
        for b in range(12, 16):                          # space: the lone grace move, echoed
            t = g.t(b, 4)
            pnote("piano_echo", t, GRACE_S, 62, 1.0)
            pnote("piano_echo", t + GRACE_S, g.steps(2) * 0.9 - GRACE_S, 64, 1.0)
        for b in range(16, 24):                          # section B: plus the octave stack
            hook_piano(b, name="piano_b", octave="piano_oct_b")
        for b in range(32, 44):                          # wall: four-note stabs from bar 41
            hook_piano(b, n_inner=3 if b >= 40 else 2, name="piano_wall",
                       octave="piano_oct_wall")
        for p, v in ((40, 0.5), (52, 0.72), (59, 0.72), (64, 0.72), (67, 0.72)):
            pnote("piano_stop", g.bar(44), g.steps(16) * 0.95, p, v,
                  {"release": 1.2, "max_len": 4.5})      # the stop chord rings into the coda
        hook_piano(45, "call", chords=False, name="piano_di")
        hook_piano(46, "answer", chords=False, name="piano_di", skip=(13,))
        pnote("piano_di", t_last, g.bar(48) - t_last, 64, 0.9, {"release": 0.4, "max_len": 8.0})

    # ---------------------------------------------------------------- rev 9: same rhythm, new shape
    v9 = rev == 9
    if v9:
        # as in v8: the guitar hook was generated above exactly as in v7, so every shared seed
        # (the fuzz's rng_x above all) draws as before; the hook guitars and body are dropped
        for k in V8_DROPPED:
            T[k] = []

        def step_of(n, dt=0.004):
            b = int(np.searchsorted(g.starts, n.t + 0.01, side="right")) - 1
            return b, min(range(16), key=lambda k: abs(g.t(b, k) + dt - n.t))

        # the wall fuzz keeps its untouched notes on steps 2, 4 and 8, which match the new right
        # hand note for note; its old steps 9, 11 and 13 would clash with the octave drop and
        # the left-hand answer, so they leave. The stop chord's fuzz stays.
        T["fuzz"] = [n for n in T["fuzz"]
                     if not (32 <= step_of(n)[0] < 44 and step_of(n)[1] in (9, 11, 13))]
        T.update({k: [] for k in V9_NAMES})
        rng_pn9 = np.random.default_rng([SEED, 9, 1])   # one timing offset per stab (R11)

        def pn9(name, t, dur, pitch, vel, params=None):
            T[name].append(Note(max(0.0, t), dur, pitch, vel, dict(params or {})))

        def hook9(b, rh=None, lo=None, hi=None, lh=None, kind=None, low_root=False):
            """The v9 hook: right-hand call (with octave stacks below and above) and the
            left-hand chord answer, on hook()'s onsets and gates."""
            kind = kind or ("call" if b % 2 == 0 else "answer")
            ch = harmony(b)
            for s, hand, top, ln, v in (CALL9 if kind == "call" else ANSWER9):
                if (rh if hand == "R" else lh) is None:
                    continue
                t = g.t(b, s) + float(np.clip(rng_pn9.normal(0.0, 0.0012), -0.002, 0.002))
                dur = g.steps(ln) * (0.9 if (s == 4 or ln >= 2) else 0.55)
                if hand == "L":
                    chord = LH9[kind][ch][s]
                    for p in chord:
                        pn9(lh, t, dur, p, v)
                    if low_root:                         # the wall's last four bars
                        pn9(lh, t, dur, LH_ROOT8[ch] - 12, 0.6 * v)
                    continue
                if s == 4:
                    # the grace dyad is struck lighter (a softer recorded layer), so its attack
                    # does not stack on the snare; the landing on E is full strength
                    (d_top, d_low), (e_top, e_low) = MOVE9
                    parts = [(t, GRACE_S, d_top, d_low, GRACE_V9 * v),
                             (t + GRACE_S, dur - GRACE_S, e_top, e_low, v)]
                else:
                    parts = [(t, dur, top, None, v)]
                for tt, dd, tp, lw, vv in parts:
                    pn9(rh, tt, dd, tp, vv)
                    if lw is not None:
                        pn9(rh, tt, dd, lw, 0.85 * vv)
                    if lo:
                        pn9(lo, tt, dd, tp - 12, vv)
                    if hi:
                        pn9(hi, tt, dd, tp + 12, vv)

        for b in range(4, 12):                           # A: right hand, octave below, left hand
            hook9(b, rh="p9_a_rh", lo="p9_a_lo", lh="p9_a_lh")
        for b in range(12, 16):                          # space: the lone chord move, echoed
            t = g.t(b, 4) + float(np.clip(rng_pn9.normal(0.0, 0.0012), -0.002, 0.002))
            (d_top, d_low), (e_top, e_low) = MOVE9
            for tp, lw, tt, dd, vv in ((d_top, d_low, t, GRACE_S, GRACE_V9),
                                       (e_top, e_low, t + GRACE_S, g.steps(2) * 0.9 - GRACE_S,
                                        1.0)):
                pn9("p9_echo", tt, dd, tp, vv)
                pn9("p9_echo", tt, dd, lw, 0.85 * vv)
        for b in range(16, 24):                          # B: plus the octave above
            hook9(b, rh="p9_b_rh", lo="p9_b_lo", hi="p9_b_hi", lh="p9_b_lh")
        for b in range(32, 44):                          # wall: full stack, low roots from 41
            hook9(b, rh="p9_w_rh", lo="p9_w_lo", hi="p9_w_hi", lh="p9_w_lh", low_root=b >= 40)
        # stop: E with no third, ringing, rolled 5 ms a note from the bottom so its attacks do
        # not stack on the stop's kick, snare and noise burst; the E2 on a softer layer
        for k, (p, v) in enumerate(((40, 0.5), (52, 0.7), (59, 0.7), (64, 0.7), (71, 0.7))):
            pn9("p9_stop", g.bar(44) + 0.005 * k, g.steps(16) * 0.95 - 0.005 * k, p, v,
                {"release": 1.2, "max_len": 4.5})
        hook9(45, rh="p9_coda_rh", kind="call")          # coda: right hand alone, dry
        hook9(46, rh="p9_coda_rh", kind="answer")
        for p in LH9["answer"]["E"][13]:                 # and the answer chord into the tape stop
            pn9("p9_coda_lh", t_last, g.bar(48) - t_last, p, 0.9,
                {"release": 0.4, "max_len": 8.0})

    # ---------------------------------------------------------------- rev 10: a new piano vamp
    v10 = rev == 10
    if v10:
        # as in v8 and v9: the guitar hook was generated above exactly as in v7, so every shared
        # seed (the fuzz's rng_x above all) draws as before; the hook guitars and body leave
        for k in V8_DROPPED:
            T[k] = []
        # the wall fuzz's v7 notes follow the old contour (onsets 4 and 8 would clash with the
        # new vamp), so they leave too; the stop chord's fuzz stays. The fuzz is then fed the
        # new top line an octave up, on its own seed.
        T["fuzz"] = [n for n in T["fuzz"] if n.t >= g.bar(44) - 0.01]
        T.update({k: [] for k in V10_NAMES})
        rng_pn10 = np.random.default_rng([SEED, 10, 1])  # one timing offset per stab (R11)
        rng_f10 = np.random.default_rng([SEED, 10, 2])   # the wall fuzz's timing

        def pn10(name, t, dur, pitch, vel, params=None):
            T[name].append(Note(max(0.0, t), dur, pitch, vel, dict(params or {})))

        def gate10(ln, grace):
            return 0.85 if ln >= 3 else (0.6 if grace is not None else 0.45)

        def vamp10(b, stab, oct_=None, hi=None, lh=None, kind=None, n_inner=2, chords=True,
                   skip=()):
            """The v10 vamp: off-beat stabs with the top line on top, close chord tones from
            the pad voicing of the bar's harmony under it (0.8x, nothing under B3), an octave-up
            and a two-octave-up copy of the top line, and the left hand's root and fifth on
            steps 2 and 9. The call's step 9 is a chord move: the D chord struck lighter, then
            50 ms later the E chord at full strength."""
            kind = kind or ("call" if b % 2 == 0 else "answer")
            ch = harmony(b)
            pcs = {nn(x) % 12 for x in PAD_VOICING6[ch]}
            for s, top, ln, v, grace in (CALL10 if kind == "call" else ANSWER10):
                if s in skip:
                    continue
                t = g.t(b, s) + float(np.clip(rng_pn10.normal(0.0, 0.0012), -0.002, 0.002))
                dur = g.steps(ln) * gate10(ln, grace)
                parts = ([(t, dur, top, v)] if grace is None else
                         [(t, GRACE_S, grace, GRACE_V10 * v),
                          (t + GRACE_S, dur - GRACE_S, top, v)])
                for tt, dd, tp, vv in parts:
                    pn10(stab, tt, dd, tp, vv)
                    if chords:
                        for c in stab_voicing(tp, pcs, n_inner):
                            if c >= STAB_FLOOR10:
                                pn10(stab, tt, dd, c, 0.8 * vv)
                    if oct_:
                        pn10(oct_, tt, dd, tp + 12, vv)
                    if hi:
                        pn10(hi, tt, dd, tp + 24, vv)
                if lh and s in (2, 9):
                    for p in LH10[ch]:
                        pn10(lh, t, dur, p, 0.8 * v)

        for b in range(4, 12):                           # A: stabs and the octave-up top line
            vamp10(b, "p10_a", oct_="p10_a_oct")
        for b in range(12, 16):                          # space: the lone grace dyad, echoed
            t = g.t(b, 9) + float(np.clip(rng_pn10.normal(0.0, 0.0012), -0.002, 0.002))
            dur = g.steps(2) * 0.6
            (d_top, d_low), (e_top, e_low) = MOVE10
            for tp, lw, tt, dd, vv in ((d_top, d_low, t, GRACE_S, GRACE_V10),
                                       (e_top, e_low, t + GRACE_S, dur - GRACE_S, 1.0)):
                pn10("p10_echo", tt, dd, tp, vv)
                pn10("p10_echo", tt, dd, lw, 0.85 * vv)
        for b in range(16, 24):                          # B: plus the sparkle and the left hand
            vamp10(b, "p10_b", oct_="p10_b_oct", hi="p10_b_hi", lh="p10_b_lh")
        for b in range(32, 44):                          # wall: full stack, 4 voices from bar 41
            vamp10(b, "p10_w", oct_="p10_w_oct", hi="p10_w_hi", lh="p10_w_lh",
                   n_inner=3 if b >= 40 else 2)
            kind = "call" if b % 2 == 0 else "answer"
            for s, top, ln, v, grace in (CALL10 if kind == "call" else ANSWER10):
                put("fuzz", b, s, ln, top + 12, 0.8 * v, gate10(ln, grace), None, 0.004,
                    rng_f10)
        # stop: E with no third, ringing, rolled 5 ms a note from the bottom
        for k, (p, v) in enumerate(((52, 0.6), (59, 0.7), (64, 0.7), (71, 0.7), (76, 0.7))):
            pn10("p10_stop", g.bar(44) + 0.005 * k, g.steps(16) * 0.95 - 0.005 * k, p, v,
                 {"release": 1.2, "max_len": 4.5})
        # coda: the top line alone, dry and mono; the answer's E5 is held into the tape stop
        vamp10(45, "p10_coda", kind="call", chords=False)
        vamp10(46, "p10_coda", kind="answer", chords=False, skip=(13,))
        pn10("p10_coda", t_last, g.bar(48) - t_last, 76, 0.9, {"release": 0.4, "max_len": 8.0})

    # ---------------------------------------------------------------- automation and tape
    length = g.bar(n_bars)
    end = length + 3.0
    t_h, t47, t48 = g.t(31, 8), g.bar(47), g.bar(48)
    hole = [(0.0, 0.0), (t_h, 0.0), (t_h + 0.02, -80.0), (t33 - 0.01, -80.0), (t33, 0.0)]
    hole_stop = hole + [(t45 - 0.02, 0.0), (t45, -80.0)]
    end_cut = [(0.0, 0.0), (t48 - 0.04, 0.0), (t48, -90.0)]
    tape_stop = ("varispeed", {"points": [(0.0, 0.0), (t47, 0.0), (t48, -1200.0)]})
    # the bridge pad sags 50 cents across bar 28 (R14); later notes start early by the lag
    t27, t28 = g.bar(27), g.bar(28)
    vs27 = [(t27, 0.0), (t27 + g.steps(12), -50.0), (t28, 0.0)]
    tt = np.linspace(t27, t28, 4001)
    lag = float(np.mean(1 - 2 ** (np.interp(tt, [p[0] for p in vs27], [p[1] for p in vs27])
                                  / 1200)) * (t28 - t27))
    for n in T["pad_bridge"]:
        if n.t >= t27 - 0.01:
            n.t -= lag

    def check(label):
        return ("c_modcheck", {"label": label, "limit_db": 1.5})

    P6 = PATCHES_V6
    amp = ("c_amp_di", P6["amp"])
    geq = ("eq", {"bands": [["peak", 220, 0.8, -3.0]]})       # the low octave is the bass's
    gsend = {"room": 0.12}
    pad_lp = ("lp_auto", {"points": lfo_points(19.3, 700.0, 2500.0, end), "q": 0.7})
    duck = ("kick", 5.0, 0.07)
    tracks = [
        # draft 1 measured the sub at 49.6 % and the limiter at 6 dB: the weight now comes from
        # a sub sine 6 dB under the oscillators and a 7 dB low shelf below 90 Hz, so the
        # bass keeps its mid growl while the sub stops eating the loudness budget
        Track("bass", "c_bass6", P6["bass"], T["bass"], gain_db=-5.0, program=38,
              inserts=[("eq", {"bands": [["lowshelf", 90, 0.7, -7.0]]}),
                       ("lp", {"f": 1200.0, "order": 2}), ("softclip", {"shave_db": 4.0}),
                       tape_stop],
              duck=duck, auto=end_cut),
        Track("grit", "c_bass6", P6["bass"], T["grit"], gain_db=-6.0, program=38,
              inserts=[("fuzz", {"gain": 7, "tone": 3000, "out": 0.5}), ("hp", {"f": 300.0}),
                       ("lp", {"f": 2500.0}), ("eq", {"bands": [["peak", 2200, 1.0, -4.0]]})],
              duck=duck),
        # draft 2: the limiter still took 5.8 dB (crest 16 dB) where the drum layers stack on
        # one instant, so the transients are shaved harder at the source (M2)
        Track("kick", "drums", P6["kit"], T["kick"], gain_db=-6.0,
              inserts=[("softclip", {"shave_db": 6.0})]),
        Track("snare", "drums", P6["kit"], T["snare"], gain_db=-5.0,
              inserts=[("softclip", {"shave_db": 6.0})]),
        Track("snare_noise", "drums", P6["kit"], T["snare_noise"], gain_db=-3.0,
              inserts=[("hp", {"f": 3000.0}), ("lp", {"f": 5000.0}),
                       ("softclip", {"shave_db": 5.0})]),
        Track("snare_sheet", "metal", P6["sheet"], T["snare_sheet"], gain_db=-8.0,
              sends={"room": 0.2}, program=14),
        Track("clap", "drums", P6["kit"], T["clap"], gain_db=-7.0, pan=-0.1,
              sends={"room": 0.15}, inserts=[("softclip", {"shave_db": 5.0})]),
        Track("hat", "drums", P6["kit"], T["hat"], gain_db=6.0, pan=0.25,
              inserts=[("hp", {"f": 7000.0}), ("softclip", {"shave_db": 3.0})]),
        Track("toms", "drums", P6["kit"], T["toms"], gain_db=-10.0, pan=0.1,
              inserts=[("softclip", {"shave_db": 4.0})]),
        Track("heart", "drums", P6["heart"], T["heart"], gain_db=-6.0),
        Track("anvil", "metal", P6["anvil"], T["anvil"], gain_db=-2.0, pan=-0.35,
              inserts=[("hp", {"f": 3500.0})], program=14),
        Track("tick", "metal", P6["tick"], T["tick"], gain_db=7.0, pan=0.4,
              inserts=[("hp", {"f": 4500.0})], program=14),
        Track("noise_hit", "drums", P6["kit"], T["noise_hit"], gain_db=-6.0, stereo={"haas": 9},
              sends={"hall": 0.2}, inserts=[("softclip", {"shave_db": 3.0})]),
        Track("rev_snare", "drums", P6["rev_snare"], T["rev_snare"], gain_db=-6.0,
              sends={"hall": 0.3}),
        # the wall's drums, crushed in parallel (R28)
        Track("kit_crush", "drums", P6["kit"], crush, gain_db=-11.0, stereo={"haas": 7},
              inserts=[("decimate", {"hold": 3, "bits": 10}), ("hp", {"f": 150.0}),
                       ("softclip", {"shave_db": 4.0})]),
        # the hook (R29)
        Track("gtr_c", "c_guitar_slide", P6["guitar"], T["gtr_c"], gain_db=-4.0, pan=-0.15,
              inserts=[amp, geq], sends=gsend, program=30),
        Track("gtr_echo", "c_guitar_slide", P6["guitar"], T["gtr_echo"], gain_db=-11.0,
              pan=-0.15, inserts=[amp, geq], sends={"echo": 0.55, "room": 0.1}, program=30),
        Track("gtr_l", "c_guitar_slide", P6["guitar"], T["gtr_l"], gain_db=-6.0, pan=-0.8,
              inserts=[amp, geq], sends=gsend, program=30),
        Track("gtr_r", "c_guitar_slide", P6["guitar"], T["gtr_r"], gain_db=-6.0, pan=0.8,
              inserts=[amp, geq], sends=gsend, program=30),
        Track("gtr_4th", "c_guitar_slide", P6["guitar"], T["gtr_4th"], gain_db=-11.0,
              inserts=[amp, geq], sends=gsend, program=30),
        Track("fuzz", "c_guitar_slide", P6["fuzz"], T["fuzz"], gain_db=-11.0,
              stereo={"haas": 12},
              inserts=[("fuzz", {"gain": 8, "tone": 5000, "out": 0.5}), ("hp", {"f": 400.0}),
                       ("eq", {"bands": [["peak", 2200, 1.0, -2.0]]}), ("lp", {"f": 6500.0})],
              sends={"room": 0.1}, program=30),
        # the intro's feedback swell and the wall's held B4 are separate tracks, so the quiet
        # intro can sit low while the wall copy keeps its level
        Track("gtr_fb", "c_guitar_slide", P6["feedback"], T["gtr_fb"], gain_db=-20.0, pan=0.3,
              inserts=[("c_amp_di", {**P6["amp"], "drive": 7.0, "di_db": None}),
                       ("lp_auto", {"points": lfo_points(7.7, 1500.0, 5000.0, end, 0.4),
                                    "q": 0.9}),
                       check("gtr_fb")],
              sends={"hall": 0.3}, program=30),
        Track("gtr_fb2", "c_guitar_slide", P6["feedback"], T["gtr_fb2"], gain_db=-17.0,
              pan=0.3,
              inserts=[("c_amp_di", {**P6["amp"], "drive": 7.0, "di_db": None}),
                       ("lp_auto", {"points": lfo_points(7.7, 1500.0, 5000.0, end, 0.4),
                                    "q": 0.9}),
                       check("gtr_fb2")],
              sends={"hall": 0.3}, program=30, auto=[(0.0, 0.0), (t45 - 0.02, 0.0),
                                                     (t45, -80.0)]),
        Track("gtr_rev", "c_guitar_slide", P6["guitar"], T["gtr_rev"], gain_db=-9.0, pan=-0.3,
              inserts=[amp], sends={"hall": 0.25}, program=30),
        # the one clean element (R24): dry and mono, then the tape stop
        Track("gtr_di", "c_guitar_slide", P6["guitar"], T["gtr_di"], gain_db=-12.0,
              inserts=[("lp", {"f": 6000.0}), tape_stop], auto=end_cut, program=27),
        # the bridge's exposed, intimate voice (R27): close and dry, light tape
        Track("keys", "c_keys_slide", P6["keys"], T["keys"], gain_db=-8.0, pan=0.05, program=4,
              inserts=[("tape", {"drive": 0.8, "wow": 0.1, "flutter": 0.0, "hf": 9000.0,
                                 "bump_db": 0.0})]),
        # textures (R13, R16, R26): none of them ducks
        Track("pad", "c_pad6", P6["pad"], T["pad"], gain_db=-20.0, program=89,
              stereo={"chorus": {"depth_ms": 6.0, "rate": 0.15, "mix": 0.35}},
              sends={"hall": 0.4}, inserts=[pad_lp, check("pad")]),
        Track("pad_wide", "c_pad6", {**P6["pad"], "a": 0.35, "r": 1.6}, T["pad_wide"],
              gain_db=-19.0, program=89,
              stereo={"chorus": {"depth_ms": 6.0, "rate": 0.15, "mix": 0.6}},
              sends={"hall": 0.4}, auto=hole_stop,
              inserts=[("lp_auto", {"points": lfo_points(19.3, 700.0, 2500.0, end, 1.1),
                                    "q": 0.7}), check("pad_wide")]),
        Track("pad_bridge", "c_pad6", P6["pad"], T["pad_bridge"], gain_db=-21.0, program=89,
              stereo={"chorus": {"depth_ms": 6.0, "rate": 0.15, "mix": 0.6}},
              sends={"hall": 0.4}, auto=hole,
              inserts=[pad_lp, ("varispeed", {"points": vs27}), check("pad_bridge")]),
        Track("strings", "c_pad6", P6["strings"], T["strings"], gain_db=-14.0, program=49,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.12, "mix": 0.6}},
              sends={"hall": 0.5}, auto=hole_stop,
              inserts=[("hp", {"f": 250.0}), check("strings")]),
        Track("drone", "c_pad6", P6["drone"], T["drone"], gain_db=-19.0, program=95,
              stereo={"chorus": {"depth_ms": 4.0, "rate": 0.09, "mix": 0.5}},
              sends={"hall": 0.15}, auto=hole_stop,
              inserts=[("tape", {"drive": 1.2, "wow": 0.1, "flutter": 0.03, "hf": 7000.0,
                                 "bump_db": 0.0}),
                       ("hp", {"f": 150.0}),
                       ("lp_auto", {"points": lfo_points(23.9, 500.0, 1600.0, end, 2.2),
                                    "q": 0.8}),
                       # the drone itself is checked; the wall's B flat ring modulation is a
                       # composed treatment whose sidebands rasp against E and B by design
                       check("drone"),
                       ("ringmod", {"freq": 233.08, "mix_points": [
                           (0.0, 0.0), (t33 - 0.02, 0.0), (t33, 0.4), (t45, 0.4)]})]),
        Track("bed", "breath", P6["bed"], T["bed"], gain_db=-11.0, program=122,
              stereo={"chorus": {"depth_ms": 3.0, "rate": 0.07, "mix": 0.7}}, auto=hole_stop,
              inserts=[("hp", {"f": 3000.0}), ("lp", {"f": 5000.0}),
                       ("decimate", {"hold": 4, "bits": 8}),
                       ("lp_auto", {"points": lfo_points(13.1, 4000.0, 11000.0, end, 0.6),
                                    "q": 0.7}),
                       check("bed")]),
    ]
    if v7:
        # (1) a thicker hook at the same level: chorus on every hook guitar, the parents
        # trimmed to make room for their unison partners, the octave layer and the body
        gch = {"chorus": {"depth_ms": 3.0, "rate": 0.7, "mix": 0.3}}
        hook_db = {"gtr_c": -7.0, "gtr_l": -9.0, "gtr_r": -9.0, "gtr_echo": -12.0,
                   "gtr_4th": -12.5}
        # (2) the background synths up 3.5 dB, with a 2 dB pocket at the guitar's cabinet peak
        back_db = {"pad": -16.1, "pad_wide": -15.1, "pad_bridge": -17.1, "drone": -15.1}
        pocket = ("eq", {"bands": [["peak", 2200, 1.0, -2.0]]})
        v7_tracks = []
        for tr in tracks:
            if tr.name in hook_db:
                tr = replace(tr, gain_db=hook_db[tr.name], stereo=gch)
            elif tr.name in back_db:
                ins = list(tr.inserts)
                if tr.name != "drone":
                    ins.insert(1, pocket)            # after the sweep, before the check
                tr = replace(tr, gain_db=back_db[tr.name], inserts=ins)
            v7_tracks.append(tr)
        amp_oct = ("c_amp_di", {**P6["amp"], "drive": 2.5, "di_db": -6.0})
        # new tracks go last, so every v6 track keeps its own noise stream
        tracks = v7_tracks + [
            Track("gtr_c_uni", "c_guitar_slide", P6["guitar"], T["gtr_c_uni"], gain_db=-8.5,
                  pan=0.15, stereo=gch, inserts=[amp, geq], sends=gsend, program=30),
            # the partners skip the bar-45 stop chord, where the limiter's peaks stack
            Track("gtr_l_uni", "c_guitar_slide", P6["guitar"], T["gtr_l_uni"], gain_db=-11.5,
                  pan=-0.65, stereo=gch, inserts=[amp, geq], sends=gsend, program=30),
            Track("gtr_r_uni", "c_guitar_slide", P6["guitar"], T["gtr_r_uni"], gain_db=-11.5,
                  pan=0.65, stereo=gch, inserts=[amp, geq], sends=gsend, program=30),
            Track("gtr_oct", "c_guitar_slide", P6["guitar"], T["gtr_oct"], gain_db=-17.0,
                  stereo=gch, inserts=[amp_oct, geq], sends=gsend, program=30),
            Track("hook_body", "c_keys_slide", PATCHES_V7["body"], T["hook_body"],
                  gain_db=-15.0, stereo={"chorus": {"depth_ms": 4.0, "rate": 0.5, "mix": 0.4}},
                  sends={"room": 0.1}, program=17,
                  inserts=[("drive", {"gain": 1.8}), ("hp", {"f": 200.0}),
                           ("lp", {"f": 2800.0}),
                           ("tape", {"drive": 0.9, "wow": 0.05, "flutter": 0.0, "hf": 9000.0,
                                     "bump_db": 0.0})]),
        ]
    if v8:
        # the early-90s house piano chain: clear of the kick and bass, a dip in the mud, a bright
        # shelf, firm compression and a shaved top, so the stabs add nothing to the limiter
        pchain = [("hp", {"f": 120.0}),
                  # (draft 4 tried a 220 Hz low shelf and a 1.2 kHz lift: the piano got louder,
                  # the untouchable stems fell 0.4 dB and the limiter rose, so it was dropped)
                  ("eq", {"bands": [["peak", 300, 1.0, -2.5], ["highshelf", 4000, 0.7, 5.0]]}),
                  ("comp", {"thresh_db": -22, "ratio": 4, "attack": 0.004, "release": 0.08}),
                  ("softclip", {"shave_db": 2.0})]
        pch = {"chorus": {"depth_ms": 2.5, "rate": 0.6, "mix": 0.2}}
        P8 = PATCHES_V8["piano"]
        # the fuzz is the one guitar left, 6 dB down, as grit under the piano
        tracks = [replace(tr, gain_db=-17.0) if tr.name == "fuzz" else tr for tr in tracks]
        db = PIANO_DB8

        def stabs(name, gain):
            # the chord stabs: centre, a little chorus and room
            return Track(name, "c_piano", P8, T[name], gain_db=gain, stereo=pch,
                         sends={"room": 0.15}, inserts=pchain, program=0)

        def octs(name, gain):
            # the classic house octave stack: top notes only, 8 dB under, Haas-widened
            return Track(name, "c_piano", P8, T[name], gain_db=gain - 8.0, stereo={"haas": 10},
                         sends={"room": 0.15}, inserts=pchain, program=0)

        tracks += [
            stabs("piano", db["A"]), stabs("piano_b", db["B"]), stabs("piano_wall", db["wall"]),
            octs("piano_oct_b", db["B"]), octs("piano_oct_wall", db["wall"]),
            stabs("piano_stop", db["stop"]),
            Track("piano_echo", "c_piano", P8, T["piano_echo"], gain_db=db["echo"],
                  sends={"echo": 0.55, "room": 0.1}, inserts=pchain, program=0),
            # the coda: bare single notes, dry and mono, then the tape stop
            Track("piano_di", "c_piano", P8, T["piano_di"], gain_db=db["coda"],
                  inserts=pchain + [tape_stop], auto=end_cut, program=0),
        ]
    if v9:
        # early-90s house piano: clear of the kick and bass, a dip in the mud, a bright shelf,
        # firm compression and a shaved top; the left hand is dipped again at 250 Hz
        # (draft 1, with v8's 4 ms attack and 2 dB shave, took the limiter to 7.5 dB: the
        # hardest sample layer's attacks stacked on the snare, so they are caught sooner here)
        comp9 = ("comp", {"thresh_db": -22, "ratio": 4, "attack": 0.002, "release": 0.08})
        clip9 = ("softclip", {"shave_db": 5.0})
        rh_chain = [("hp", {"f": 120.0}),
                    ("eq", {"bands": [["peak", 300, 1.0, -3.0], ["highshelf", 4000, 0.7, 5.0]]}),
                    comp9, clip9]
        # the left hand: its C3 and D3 roots leave the low octave to the bass (draft 1 measured
        # it at -12 dB below 150 Hz), with a deeper dip at 250 Hz
        lh_chain = [("hp", {"f": 150.0}),
                    ("eq", {"bands": [["peak", 300, 1.0, -3.0], ["peak", 250, 1.0, -4.0],
                                      ["highshelf", 4000, 0.7, 5.0]]}),
                    comp9, clip9]
        pch9 = {"chorus": {"depth_ms": 2.5, "rate": 0.6, "mix": 0.2}}
        P9 = PATCHES_V9["piano"]
        # the fuzz is the one guitar left, 6 dB down, as grit under the piano
        tracks = [replace(tr, gain_db=-17.0) if tr.name == "fuzz" else tr for tr in tracks]
        db = PIANO_DB9

        def rh9(name, gain):
            return Track(name, "c_piano", P9, T[name], gain_db=gain, stereo=pch9,
                         sends={"room": 0.15}, inserts=rh_chain, program=0)

        def lo9(name, gain):
            # the octave-below stack, centred, 6 dB under the right hand
            return Track(name, "c_piano", P9, T[name], gain_db=gain - 6.0,
                         sends={"room": 0.15}, inserts=rh_chain, program=0)

        def hi9(name, gain):
            # the octave-above stack, 8 dB under and Haas-widened
            return Track(name, "c_piano", P9, T[name], gain_db=gain - 8.0, stereo={"haas": 10},
                         sends={"room": 0.15}, inserts=rh_chain, program=0)

        def lh9(name, gain):
            return Track(name, "c_piano", P9, T[name], gain_db=gain, stereo=pch9,
                         sends={"room": 0.08}, inserts=lh_chain, program=0)

        tracks += [
            rh9("p9_a_rh", db["A_rh"]), lo9("p9_a_lo", db["A_rh"]), lh9("p9_a_lh", db["A_lh"]),
            Track("p9_echo", "c_piano", P9, T["p9_echo"], gain_db=db["echo"],
                  sends={"echo": 0.55, "room": 0.1}, inserts=rh_chain, program=0),
            rh9("p9_b_rh", db["B_rh"]), lo9("p9_b_lo", db["B_rh"]), hi9("p9_b_hi", db["B_rh"]),
            lh9("p9_b_lh", db["B_lh"]),
            rh9("p9_w_rh", db["wall_rh"]), lo9("p9_w_lo", db["wall_rh"]),
            hi9("p9_w_hi", db["wall_rh"]), lh9("p9_w_lh", db["wall_lh"]),
            rh9("p9_stop", db["stop"]),
            # the coda: dry and mono, then the tape stop
            Track("p9_coda_rh", "c_piano", P9, T["p9_coda_rh"], gain_db=db["coda_rh"],
                  inserts=rh_chain + [tape_stop], auto=end_cut, program=0),
            Track("p9_coda_lh", "c_piano", P9, T["p9_coda_lh"], gain_db=db["coda_lh"],
                  inserts=lh_chain + [tape_stop], auto=end_cut, program=0),
        ]
    if v10:
        # the early-90s house piano, voiced high and bright: out of the bass's register (200 Hz
        # high-pass, 4 dB out of 300 Hz), a 2.8 kHz bite and a 4 kHz shelf, firm fast
        # compression and a shaved top
        comp10 = ("comp", {"thresh_db": -22, "ratio": 4, "attack": 0.002, "release": 0.06})
        clip10 = ("softclip", {"shave_db": 5.0})
        # (draft 2, with +3 dB at 2.8 kHz and a 4 dB shelf, had the piano's fundamentals pulling
        # the untouched kick 0.3 dB down against the mix while the upper mids stayed short, so
        # the weight moves up: more bite and shelf, and the stabs 1 dB lower)
        bite10 = ("eq", {"bands": [["peak", 300, 1.0, -4.0], ["peak", 2800, 0.9, 5.0],
                                   ["highshelf", 4000, 0.7, 5.0]]})
        p_chain = [("hp", {"f": 200.0}), bite10, comp10, clip10]
        lh_chain10 = [("hp", {"f": 220.0}), bite10, comp10, clip10]
        pch10 = {"chorus": {"depth_ms": 2.5, "rate": 0.6, "mix": 0.2}}
        P10 = PATCHES_V10["piano"]
        # the fuzz is the one guitar left, quietly under the piano, playing the new top line
        tracks = [replace(tr, gain_db=-17.0) if tr.name == "fuzz" else tr for tr in tracks]
        db = PIANO_DB10

        def stab10(name, gain):
            # the stabs: centred, a little chorus and room
            return Track(name, "c_piano", P10, T[name], gain_db=gain, stereo=pch10,
                         sends={"room": 0.12}, inserts=p_chain, program=0)

        def oct10(name, gain):
            # the top line an octave up, about 4-5 dB under in the mid band, Haas-widened
            return Track(name, "c_piano", P10, T[name], gain_db=gain + OCT_DB10,
                         stereo={"haas": 10},
                         sends={"room": 0.12}, inserts=p_chain, program=0)

        def hi10(name, gain):
            # the top line two octaves up, about 12 dB under in the mid band: sparkle
            return Track(name, "c_piano", P10, T[name], gain_db=gain + HI_DB10,
                         stereo={"haas": 10},
                         sends={"room": 0.12}, inserts=p_chain, program=0)

        def lh10(name, gain):
            # the left hand's root and fifth, 6 dB under
            return Track(name, "c_piano", P10, T[name], gain_db=gain - 6.0, stereo=pch10,
                         sends={"room": 0.08}, inserts=lh_chain10, program=0)

        tracks += [
            stab10("p10_a", db["A"]), oct10("p10_a_oct", db["A"]),
            Track("p10_echo", "c_piano", P10, T["p10_echo"], gain_db=db["echo"],
                  sends={"echo": 0.55, "room": 0.1}, inserts=p_chain, program=0),
            stab10("p10_b", db["B"]), oct10("p10_b_oct", db["B"]), hi10("p10_b_hi", db["B"]),
            lh10("p10_b_lh", db["B"]),
            stab10("p10_w", db["wall"]), oct10("p10_w_oct", db["wall"]),
            hi10("p10_w_hi", db["wall"]), lh10("p10_w_lh", db["wall"]),
            stab10("p10_stop", db["stop"]),
            # the coda: the top line alone, dry and mono, then the tape stop
            Track("p10_coda", "c_piano", P10, T["p10_coda"], gain_db=db["coda"],
                  inserts=p_chain + [tape_stop], auto=end_cut, program=0),
        ]
    # the bar-32 hole is total silence apart from the reversed swell: the room and hall
    # returns are closed for its half bar only
    hole_gate = ("gate", {"windows": [(0.0, t_h), (t33 - 0.003, end + 10.0)], "smooth": 0.01})
    buses = {
        "room": {"type": "reverb", "ir": {"decay": 0.6, "predelay": 0.006, "tone": 7000,
                                          "size": 0.5}, "return_db": -8, "inserts": [hole_gate]},
        "hall": {"type": "reverb", "ir": {"decay": 4.5, "predelay": 0.03, "tone": 6000,
                                          "size": 1.6}, "return_db": -5, "inserts": [hole_gate]},
        "echo": {"type": "echo", "echo": {"time_s": g.steps(3), "feedback": 0.45, "repeats": 6,
                                          "tone": 3000, "spread": 0.6, "wow": 0.3, "sat": 0.2},
                 "return_db": -4},
    }
    master = {"target_lufs": -14.0, "ceiling": -1.5, "tail": 3.0, "linear_phase_mono": True,
              "mono_below": 150.0, "eq": [["hp", 28, 0.7]], "true_peak": True}
    sections = [(name, g.bar(a), g.bar(b)) for name, (a, b) in sec.items()]
    return Song("Concrete Tide", "c_concrete_tide", BPM, length, tracks, buses, master, sections,
                SEED, rev, note_text,
                meta={"bar_starts": [g.bar(i) for i in range(n_bars + 1)]})
