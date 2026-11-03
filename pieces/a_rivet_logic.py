"""A. Rivet Logic: angular electronic funk, 112 BPM, E with a contested third.

Revision history lives in build(): later revisions are gated on `rev`, so every version stays
reproducible from this one file.
"""
import numpy as np

from engine.dsp import undb
from engine.rules import Grid, euclid, rotate, thin, nudge, syncopation
from engine.song import Song, Track
from pieces.common import add_cell, add_hits, add_line, nn

BPM = 112
SEED = 1101

# R1/R2 bass cells: (step, length, semitones above root, velocity)
BASS_E1 = [(0, 2, 0, 1.0), (3, 1, 12, 0.8), (6, 1, 10, 0.75), (7, 1, 0, 0.9),
           (10, 2, 3, 0.85), (13, 1, 5, 0.8), (14, 1, 7, 0.7)]
BASS_E2 = [(0, 2, 0, 1.0), (3, 1, 12, 0.8), (6, 1, 10, 0.75), (7, 1, 0, 0.9),
           (10, 1, 3, 0.85), (11, 1, 4, 0.6), (12, 1, 7, 0.8), (14, 1, 10, 0.75)]
BASS_A = [(0, 3, 0, 1.0), (4, 1, 12, 0.7), (6, 1, 0, 0.8), (9, 1, 7, 0.85),
          (11, 1, 10, 0.8), (12, 2, 12, 0.7), (15, 1, 9, 0.6)]
BASS_BRIDGE = [(0, 5, 0, 1.0), (6, 1, 12, 0.7), (8, 6, 7, 0.8), (15, 1, 0, 0.6)]

# R4/R5 clipped motif, 3+3+2 placement: (step, length, pitch)
HOOK_Q = [(0, 1, "B4"), (3, 1, "D5"), (6, 2, "E5"), (14, 1, "G5")]
HOOK_A = [(0, 1, "B4"), (3, 1, "D5"), (6, 2, "A4")]

LEAD_B = [(0, 0, 6, "E5"), (0, 6, 2, "D5"), (0, 8, 6, "G5"), (0, 14, 2, "E5"),
          (1, 0, 10, "D5"), (1, 12, 4, "C5"), (2, 0, 4, "A4"), (2, 4, 2, "C5"),
          (2, 6, 8, "D5"), (2, 14, 2, "E5"), (3, 0, 12, "A4")]
LEAD_B_END2 = [(3, 0, 4, "G5"), (3, 4, 12, "E5")]

# rev 3 (R22): the third leaves the bass; F# on the strong off-beat gives a sus2 colour.
# Onsets are unchanged, so the variation draws (nudge/thin) are the same as in rev 2.
BASS_E1_3 = [(0, 2, 0, 1.0), (3, 1, 12, 0.8), (6, 1, 10, 0.75), (7, 1, 0, 0.9),
             (10, 2, 2, 0.85), (13, 1, 5, 0.8), (14, 1, 7, 0.7)]
BASS_E2_3 = [(0, 2, 0, 1.0), (3, 1, 12, 0.8), (6, 1, 10, 0.75), (7, 1, 0, 0.9),
             (10, 1, 2, 0.85), (11, 1, 5, 0.6), (12, 1, 7, 0.8), (14, 1, 10, 0.75)]
BASS_HELD = [(0, 14, 0, 0.8)]                    # rev 3 bridge: one held root per bar
# rev 3 (R4/R5): the hook lives in the bass's gaps, a 3+3+3 pickup locked to the guitar's 2 and 5
HOOK_Q3 = [(2, 1, "B4"), (5, 1, "D5"), (8, 2, "E5"), (15, 1, "G5")]
HOOK_A3 = [(2, 1, "B4"), (5, 1, "D5"), (9, 2, "A4")]
# rev 3 (R12): a fixed kick map per section
KICK_GROOVE = [(0, 1.0), (7, 0.8), (10, 0.9)]
KICK_DISPLACE = [(0, 1.0), (3, 0.7), (7, 0.8), (10, 0.9)]
KICK_LIFT = [(0, 1.0), (6, 0.8), (10, 0.9), (13, 0.75)]
METAL_LOOP = euclid(5, 13)                       # rev 3 (R8/R16): 5-in-13 over the running 16ths

PATCHES = {
    "drums": {
        "kick": {"f_start": 150, "f_end": 47, "p_tau": 0.025, "decay": 0.28, "click": 0.35,
                 "drive": 1.8},
        "snare": {"f1": 190, "f2": 340, "decay": 0.13, "body_decay": 0.05, "noise_f": 4200,
                  "noise": 1.0, "body": 0.7},
        "snare_drop": {"decay": 0.35, "drop": 9, "noise_f": 2800},
        "hat": {"decay": 0.035, "tune": 1.7, "bp": 10000},
        "hat_open": {"open_decay": 0.22, "tune": 1.7},
        "clap": {"f": 1400, "tail": 0.09},
        "rim": {"f": 1900},
        "tom_low": {"decay": 0.5, "sweep": 1.2, "fall": 7},
        "tom_mid": {"decay": 0.4, "sweep": 1.2, "fall": 7},
        "tom_high": {"decay": 0.35, "sweep": 1.2, "fall": 7},
    },
    "bass": {"saw": 0.75, "pulse": 0.55, "pw": 0.28, "sub": 0.55, "detune": 0.06,
             "cutoff": 220, "fenv": 2600, "fdecay": 0.09, "res": 0.45, "drive": 2.0,
             "a": 0.001, "d": 0.15, "s": 0.6, "r": 0.025},
    "hook": {"pw": 0.3, "pwm": 0.15, "pwm_rate": 0.21, "cutoff": 1800, "fenv": 6000,
             "fdecay": 0.04, "q": 3.0, "d": 0.05, "s": 0.35, "oct": 0.25},
    "guitar": {"t60": 1.4, "bright": 0.75, "pick": 0.12, "ring": 0.03, "drive": 1.5,
               "body_lp": 3800, "hp": 180},
    "stab": {"ratio": 1.0, "index": 2.2, "idecay": 0.12, "a": 0.003, "d": 0.25, "s": 0.0,
             "r": 0.12, "ratio2": 3.0, "index2": 0.8, "lp": 5500},
    "lead": {"saw": 0.3, "sq": 0.7, "a": 0.01, "s": 0.8, "r": 0.15, "cutoff": 2600, "q": 1.6,
             "vib": 0.18, "vib_delay": 0.35, "vib_rate": 5.6, "glide": 0.05, "drive": 1.4},
    "organ": {"bars": [[1, 1.0], [2, 0.35], [3, 0.5], [4, 0.1], [6, 0.08]], "a": 0.04,
              "r": 0.3, "flutter": 0.003},
    "fog": {"a": 1.5, "r": 2.5, "s": 0.9, "voices": 4, "spread": 0.18, "cutoff": 900,
            "move_oct": 1.4, "move_rate": 0.25, "formant": "o", "formant_morph": 0.6},
}


def bass_cell(b, rng, vary, v3=False):
    """E cells alternate; from rev 2, every fourth bar mutates by one move (R3)."""
    if v3:
        cell = BASS_E1_3 if b % 2 == 0 else BASS_E2_3
    else:
        cell = BASS_E1 if b % 2 == 0 else BASS_E2
    if vary and b % 4 == 3:
        cell = nudge(cell, rng) if rng.uniform() < 0.6 else thin(cell, rng)
    return cell


def build(rev=1):
    # the realisation is fixed to revision 1 so later revisions change only what they say
    rng = np.random.default_rng([SEED, 1])
    v2 = rev >= 2
    v3 = rev >= 3
    vrng = np.random.default_rng([SEED, 99])      # variation choices, stable across revisions
    rng3 = np.random.default_rng([SEED, 3])       # rev 3's new jittered parts draw from here
    overrides = {32: 8}                     # one 2/4 bar: the angular jolt after the break
    g = Grid(BPM, swing=0.54, overrides=overrides)
    sec = {"intro": (0, 4), "groove": (4, 12), "hook": (12, 20), "displace": (20, 28),
           "break": (28, 33), "lift_A": (33, 41), "return": (41, 49), "bridge": (49, 55),
           "final": (55, 67), "end": (67, 68)}
    n_bars = 68
    dr, bass, hook, gtr, stab, lead, org, fog = ([] for _ in range(8))
    met, stab_dec = [], []                        # rev 3 only
    HQ, HA = (HOOK_Q3, HOOK_A3) if v3 else (HOOK_Q, HOOK_A)

    def hook_pitch(p, major=False):
        """R22: the hook's third is G5 (minor) or, in the 'major light', G#5."""
        return nn(p) + (1 if major and p == "G5" else 0)

    def drums_groove(b, var=0, hats=True, snare=True, kick=True, open_hat=None,
                     kick_map=None, hat_div=1, snare_gm=38):
        if kick:
            km = kick_map if kick_map is not None else [(0, 1.0), (7, 0.8), (10, 0.9)]
            add_hits(dr, g, b, km + ([(14, 0.55)] if var else []), 36)
        if snare:
            add_hits(dr, g, b, [(4, 0.95), (12, 1.0)], snare_gm)
            add_hits(dr, g, b, [(15, 0.22)], 38)
        if hats:
            acc = [0.85, 0.35, 0.6, 0.4]
            steps = [(s, acc[s % 4]) for s in range(0, 16, hat_div)
                     if not (open_hat and s in (14, 15))]
            add_hits(dr, g, b, steps, 42, jitter=0.002, rng=rng)
            if open_hat:
                add_hits(dr, g, b, [(14, 0.6)], 46)

    # ---------------------------------------------------------------- intro (drums only)
    for b in range(*sec["intro"]):
        add_hits(dr, g, b, [(0, 1.0), (7, 0.8), (10, 0.9)], 36)
        add_hits(dr, g, b, [(s, [0.8, 0.3, 0.55, 0.35][s % 4]) for s in range(16)], 42,
                 jitter=0.002, rng=rng)
        if b == 3:
            add_hits(dr, g, b, [(4, 0.9), (12, 1.0)], 38)
            add_hits(dr, g, b, [(13, 0.7), (14, 0.8), (15, 0.9)], 48)

    # ---------------------------------------------------------------- groove / hook / displace
    E2 = nn("E2")
    for b in range(sec["groove"][0], sec["displace"][1]):
        if v3:
            # rev 3 (R12): eighth hats in the groove, sixteenths from the hook, and a kick that
            # pushes on 3 against the walking hook in the displacement
            km = KICK_DISPLACE if b >= sec["displace"][0] else KICK_GROOVE
            drums_groove(b, var=(b % 4 == 3), open_hat=(b % 2 == 1), kick_map=km,
                         hat_div=2 if b < sec["hook"][0] else 1)
            # rev 3 (R15): the bass filter opens across the first act, 0.7 to 1.3
            bp = {"bright": 0.7 + 0.6 * (b - sec["groove"][0]) / (sec["displace"][1] - 1 - 4)}
        else:
            drums_groove(b, var=(b % 4 == 3), open_hat=(b % 2 == 1))
            bp = None
        add_cell(bass, g, b, bass_cell(b, vrng, v2 and b >= sec["hook"][0], v3), E2, gate=0.62,
                 params=bp)
        if v2 and 8 <= b < sec["hook"][0]:
            # rev 2: the hook is foreshadowed, dark and quiet, before it arrives
            motif = HQ if b % 2 == 0 else HA
            for s, l, p in motif:
                add_cell(hook, g, b, [(s, l, 0, 0.55)], nn(p), gate=0.5,
                         params={"bright": 0.35 + 0.15 * (b - 8)})

    for b in range(sec["hook"][0], sec["displace"][1]):
        # muted rhythm guitar as percussion (R6): pitched accents on 2,5,10,13, scratches between
        G3, D4 = nn("G3"), nn("D4")
        for s in (2, 5, 10, 13):
            for p in (G3, D4):
                add_cell(gtr, g, b, [(s, 1, 0, 0.85)], p, gate=0.45, jitter=0.002, rng=rng,
                         params={"mute": 0.55})
        for s in (3, 7, 11, 15):
            add_cell(gtr, g, b, [(s, 1, 0, 0.35)], G3, gate=0.3, jitter=0.002, rng=rng,
                     params={"mute": 1.0})

    for b in range(sec["hook"][0], sec["displace"][1]):
        motif = HQ if b % 2 == 0 else HA
        shift = 0
        if b >= sec["displace"][0]:
            shift = (b - sec["displace"][0]) + 1      # R5: walks one 16th per bar
        cell = rotate([(s, l, nn(p), 0.9) for s, l, p in motif], shift)
        for s, l, p, v in cell:
            add_cell(hook, g, b, [(s, l, 0, v)], p, gate=0.45)

    if v3:
        # rev 3 (R21): a quiet quartal upper structure over the pedal before the organ arrives,
        # E-A-D alternating with D-G-C on step 10 of every second bar, 6 dB under the stab fader
        for i, b in enumerate(range(sec["hook"][0] + 1, sec["displace"][0], 2)):
            chord = ("E4", "A4", "D5") if i % 2 == 0 else ("D4", "G4", "C5")
            for p in chord:
                add_cell(stab, g, b, [(10, 2, 0, 0.8)], nn(p), gate=0.5,
                         params={"gain": undb(-6.0)})
    for b in range(sec["displace"][0], sec["displace"][1], 2):
        for p in ("E4", "A4", "D5"):                  # R21 quartal stab
            # rev 3: the displacement stabs go to their own decimated track (R14)
            add_cell(stab_dec if v3 else stab, g, b + 1, [(10, 2, 0, 0.8)], nn(p), gate=0.5)

    # ---------------------------------------------------------------- break (R19 subtraction)
    for b in range(sec["break"][0], sec["break"][0] + 4):
        add_hits(dr, g, b, [(0, 1.0)], 36)
        add_hits(dr, g, b, [(4, 0.6), (10, 0.5), (12, 0.7)], 37)
        for p in ("G3", "D4"):
            add_cell(gtr, g, b, [(2, 1, 0, 0.9), (10, 1, 0, 0.8)], nn(p), gate=0.45,
                     params={"mute": 0.5})
        cell = rotate([(s, l, nn(p), 0.8) for s, l, p in HQ], 2 * (b - 28))
        for s, l, p, v in cell:
            add_cell(hook, g, b, [(s, l, 0, v)], p, gate=0.4)
    b = 32                                           # the 2/4 bar: dropped-pitch snare fill
    add_hits(dr, g, b, [(0, 0.9), (2, 0.7), (4, 1.0), (6, 0.8), (7, 0.9)], 40)
    add_hits(dr, g, b, [(4, 0.8), (6, 0.9)], 41)

    # ---------------------------------------------------------------- lift to A
    A1 = nn("A1")
    for b in range(*sec["lift_A"]):
        if v3:
            # rev 3 (R10, R12): the lift's own kick map and the big dropped snare on the backbeat
            drums_groove(b, open_hat=(b % 2 == 1), kick_map=KICK_LIFT, snare_gm=40)
        else:
            drums_groove(b, var=(b % 4 == 3), open_hat=(b % 2 == 1))
        add_hits(dr, g, b, [(4, 0.7), (12, 0.8)], 39)
        if v3:
            # rev 3 (R15): the bass resets darker and slides up into its octave pops, A only
            for s, l, off, v in BASS_A:
                bp = {"bright": 0.8}
                if off == 12:
                    bp["from_pitch"] = A1
                add_cell(bass, g, b, [(s, l, off, v)], A1, gate=0.6, params=bp)
            # the guitar keeps running, moved to A (pitched A3/E4 accents, A3 scratches)
            for s in (2, 5, 10, 13):
                for p in ("A3", "E4"):
                    add_cell(gtr, g, b, [(s, 1, 0, 0.85)], nn(p), gate=0.45, jitter=0.002,
                             rng=rng3, params={"mute": 0.55})
            for s in (3, 7, 11, 15):
                add_cell(gtr, g, b, [(s, 1, 0, 0.35)], nn("A3"), gate=0.3, jitter=0.002,
                         rng=rng3, params={"mute": 1.0})
            if b >= sec["lift_A"][0] + 4:
                # the hook arrives halfway through the lift, a fifth down in A
                for s, l, p in (HQ if b % 2 == 0 else HA):
                    add_cell(hook, g, b, [(s, l, 0, 0.9)], nn(p) - 7, gate=0.45)
        else:
            add_cell(bass, g, b, BASS_A, A1, gate=0.6)
    k = sec["lift_A"][0]
    add_line(lead, g, k, LEAD_B, gate=0.9)
    add_line(lead, g, k + 4, LEAD_B[:-1] + LEAD_B_END2, gate=0.9)

    # ---------------------------------------------------------------- return to E
    for b in range(*sec["return"]):
        drums_groove(b, var=(b % 4 == 3), open_hat=(b % 2 == 1))
        add_cell(bass, g, b, bass_cell(b, vrng, v2, v3), E2, gate=0.62)
        motif = HQ if b % 2 == 0 else HA
        for s, l, p in motif:
            # rev 3 (R22): the return is the major light, so the hook's third is G#5
            add_cell(hook, g, b, [(s, l, 0, 0.9)], hook_pitch(p, major=v3), gate=0.45)
        # rev 3: the guitar dyad drops its G natural for B/F# under the major light
        dyad = ("B3", "F#4") if v3 else ("G3", "D4")
        for s in (2, 5, 10, 13):
            for p in dyad:
                add_cell(gtr, g, b, [(s, 1, 0, 0.85)], nn(p), gate=0.45, jitter=0.002, rng=rng,
                         params={"mute": 0.55})
        # planing fourths; rev 3 replaces D-G-C with C#-F#-B, a bright 6/9 shape over the pedal
        second = ("C#4", "F#4", "B4") if v3 else ("D4", "G4", "C5")
        chord = ("E4", "A4", "D5") if b % 2 == 0 else second
        for p in chord:
            add_cell(org, g, b, [(0, 16, 0, 0.6)], nn(p), gate=0.97)

    # ---------------------------------------------------------------- bridge: C / F# (tritone)
    for i, b in enumerate(range(*sec["bridge"])):
        root = nn("C2") if i % 2 == 0 else nn("F#1")
        bars_left = sec["bridge"][1] - b               # 6..1
        if v3:
            # rev 3: the bridge is a valley. One held root per bar; the kick leaves for the last
            # two bars and the bass for the last one, so bar 55's downbeat is a return (R19).
            # synth_bass ignores velocity, so a held note needs a per-note gain to sit low
            if bars_left > 1:
                add_cell(bass, g, b, BASS_HELD, root, gate=1.0,
                         params={"bright": 0.55, "gain": undb(-9.0)})
            if bars_left > 2:
                add_hits(dr, g, b, [(0, 0.85), (10, 0.6)], 36)
            kv = 0.75                                    # the kit plays softer in the valley
        else:
            add_cell(bass, g, b, BASS_BRIDGE, root, gate=0.85)
            add_hits(dr, g, b, [(0, 1.0), (10, 0.7)], 36)
            kv = 1.0
        add_hits(dr, g, b, [(12, 0.9 * kv)], 40)
        toms = [41, 45, 48]
        add_hits(dr, g, b, [(s, 0.6 * kv) for s in (3, 6, 9, 13)], toms[i % 3])
        chord = ("C3", "G3", "D4") if i % 2 == 0 else ("F#3", "C#4", "G#4")
        for p in chord:
            add_cell(fog, g, b, [(0, 16, 0, 0.7)], nn(p), gate=1.0)

    # ---------------------------------------------------------------- final + subtraction
    f0 = sec["final"][0]
    for b in range(*sec["final"]):
        last = sec["final"][1] - b                     # 12..1
        # rev 3 (R22): major and minor forms alternate by two-bar phrase, major first, and the
        # last hook statement (bar 63) is major
        major = v3 and ((b - f0) // 2) % 2 == 0
        if v3:
            # the final's kit is the union of the groove and displacement maps
            drums_groove(b, var=(b % 4 == 3), hats=(last > 2), open_hat=(b % 2 == 1 and last > 2),
                         kick_map=KICK_DISPLACE)
            if f0 + 4 <= b < f0 + 8:
                add_hits(dr, g, b, [(4, 0.55), (12, 0.65)], 39)
        else:
            drums_groove(b, var=(b % 4 == 3), hats=(last > 2), open_hat=(b % 2 == 1 and last > 2))
        add_cell(bass, g, b, bass_cell(b, vrng, v2, v3), E2, gate=0.62)
        if last > 3:
            motif = HQ if (b % 2 == 0 or (v3 and last == 4)) else HA
            for s, l, p in motif:
                add_cell(hook, g, b, [(s, l, 0, 0.9)], hook_pitch(p, major), gate=0.45)
        if last > 4:
            dyad = ("B3", "F#4") if major else ("G3", "D4")
            for s in (2, 5, 10, 13):
                for p in dyad:
                    add_cell(gtr, g, b, [(s, 1, 0, 0.85)], nn(p), gate=0.45, jitter=0.002,
                             rng=rng, params={"mute": 0.55})
        if last > 6 and b % 2 == 1:
            for p in ("E4", "A4", "D5"):
                add_cell(stab, g, b, [(10, 2, 0, 0.8)], nn(p), gate=0.5)
        if v3 and b < f0 + 8:
            # rev 3: the return's organ planing, 3 dB under the organ fader, following the phrase
            second = ("C#4", "F#4", "B4") if major else ("D4", "G4", "C5")
            for p in (("E4", "A4", "D5") if b % 2 == 0 else second):
                add_cell(org, g, b, [(0, 16, 0, 0.6)], nn(p), gate=0.97,
                         params={"gain": undb(-3.0)})
    if v3:
        # rev 3: the lead runs through bars 57-62 over the hook, so bars 59-62 are the one place
        # where everything plays. Bar 59 sits in a major phrase, so its C5 becomes C#5.
        full = [(bo, s, ln, "C#5" if (bo == 2 and p == "C5") else p) for bo, s, ln, p in LEAD_B]
        add_line(lead, g, f0 + 2, full, gate=0.9, vel=0.75)
        add_line(lead, g, f0 + 6, LEAD_B[:4] + [(1, 0, 12, "E5")], gate=0.9, vel=0.8)
    elif v2:
        # rev 2: the A-section melody comes back over E, so the lift is remembered
        add_line(lead, g, f0 + 2, LEAD_B[:5], gate=0.9, vel=0.75)
        add_line(lead, g, f0 + 6, LEAD_B[:4] + [(1, 0, 12, "E5")], gate=0.9, vel=0.8)
    if v2:
        # section-end fills: dropped snare and a tom fall on the last beat
        for b in (11, 19, 27, 40, 48):
            add_hits(dr, g, b, [(12, 0.85), (14, 0.7), (15, 0.9)], 40)
            add_hits(dr, g, b, [(13, 0.7), (14, 0.8)], 45)
            add_hits(dr, g, b, [(15, 0.8)], 41)
    b = sec["end"][0]
    add_hits(dr, g, b, [(0, 1.0)], 36)
    add_hits(dr, g, b, [(0, 1.0)], 39)
    add_cell(bass, g, b, [(0, 3, 0, 1.0)], nn("E1"), gate=1.0)

    if v3:
        # rev 3 (R8, R16): a tuned metal voice plays 5-in-13 over the running sixteenth counter,
        # so it meets the bar line the same way only every 13 bars. In from the displacement,
        # out for the break and the bridge, back at bar 55, out with the hats.
        on = (list(range(sec["displace"][0], sec["displace"][1]))
              + list(range(sec["lift_A"][0], sec["return"][1]))
              + list(range(sec["final"][0], sec["final"][1] - 2)))
        for b in on:
            s0 = int(round(g.starts[b] / g.step_s))
            for s in range(g.lengths[b]):
                k = (s0 + s) % 13
                if METAL_LOOP[k]:
                    add_hits(met, g, b, [(s, 0.75 if k == 0 else 0.6)], nn("B5"))

    length = g.bar(n_bars)
    if v2:
        return build_mix_v2(rev, g, n_bars, sec, length, dr, bass, hook, gtr, stab, lead, org, fog,
                            met=met, stab_dec=stab_dec)
    tracks = [
        Track("kick", "drums", PATCHES["drums"], [n for n in dr if n.pitch == 36], gain_db=-1.0),
        Track("kit", "drums", PATCHES["drums"], [n for n in dr if n.pitch != 36], gain_db=-2.0),
        Track("bass", "synth_bass", PATCHES["bass"], bass, gain_db=-1.5, duck=("kick", 3.0, 0.06),
              program=38),
        Track("hook", "blip", PATCHES["hook"], hook, gain_db=-9.0, pan=-0.3,
              sends={"slap": 0.5}, program=80),
        Track("guitar", "pluck", PATCHES["guitar"], gtr, gain_db=-13.0, pan=0.38, program=28,
              inserts=[("eq", {"bands": [["peak", 1500, 1.2, 3.0]]})]),
        Track("stab", "fm", PATCHES["stab"], stab, gain_db=-10.0,
              stereo={"chorus": {"depth_ms": 2.0, "rate": 0.8, "mix": 0.5}}, sends={"slap": 0.3},
              program=61),
        Track("lead", "lead", PATCHES["lead"], lead, gain_db=-10.0, pan=0.1,
              sends={"echo": 0.35, "plate": 0.12}, program=81),
        Track("organ", "organ", PATCHES["organ"], org, gain_db=-17.0,
              stereo={"chorus": {"depth_ms": 4.0, "rate": 0.35, "mix": 0.6}}, program=17,
              inserts=[("tape", {"drive": 1.2, "wow": 0.6, "flutter": 0.4, "hf": 7000})]),
        Track("fog", "pad", PATCHES["fog"], fog, gain_db=-13.0,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.2, "mix": 0.7}},
              sends={"plate": 0.6}, program=91),
    ]
    buses = {
        "slap": {"type": "echo", "echo": {"time_s": g.steps(3), "feedback": 0.35, "repeats": 3,
                                          "tone": 4000, "spread": 0.8}, "return_db": -4},
        "echo": {"type": "echo", "echo": {"time_s": g.steps(6), "feedback": 0.45, "repeats": 5,
                                          "tone": 3000, "spread": 0.6, "wow": 1.0, "sat": 0.5},
                 "return_db": -5},
        "plate": {"type": "reverb", "ir": {"decay": 2.2, "predelay": 0.02, "tone": 6500},
                  "return_db": -6},
    }
    master = {"target_lufs": -14.0, "ceiling": -1.5, "tail": 2.5,
              "eq": [["hp", 28, 0.7], ["peak", 320, 1.0, -1.0]],
              "comp": {"thresh_db": -16, "ratio": 1.6, "attack": 0.02, "release": 0.2},
              "tape": {"drive": 0.6, "wow": 0.06, "flutter": 0.08, "hiss_db": -80, "hf": 15500,
                       "bump_db": 1.0}}
    sections = [(name, g.bar(a), g.bar(b)) for name, (a, b) in sec.items()]
    notes = {1: "First complete arrangement."}
    return Song("Rivet Logic", "a_rivet_logic", BPM, length, tracks, buses, master, sections,
                SEED, rev, notes.get(rev, ""),
                meta={"bar_starts": [g.bar(i) for i in range(n_bars + 1)],
                      "bass_syncopation": [syncopation(BASS_E1), syncopation(BASS_E2),
                                           syncopation(BASS_A)]})


REV_NOTES = {
    2: "Mix rebalanced toward the mids (bass -2.5 dB and less sub, hook +3.5 dB and brighter, "
       "guitar +3.5 dB, hats on their own panned track +4 dB, presence shelf). Lows made truly "
       "mono. Bass cell mutates every fourth bar from the hook on (R3). The hook is "
       "foreshadowed, dark, in the four bars before it arrives. Section-end fills. The lift's "
       "melody returns over E in the final section.",
    3: "Written by the agentic loop's producer from a critic's reading of v2; the note itself was "
       "written afterwards by Claude Opus 5.5 from the code. The hook moves into the bass's gaps "
       "(steps 2, 5, 8/9, 15), a 3+3+3 pickup locked to the guitar's accents instead of doubling "
       "the bass, so the displacement walk can be heard. The third leaves the bass (F sharp on "
       "the strong off-beat). Each section has its own kick map; eighth hats in the groove, "
       "sixteenths from the hook. The bass filter opens across the first act. A quiet quartal "
       "upper structure appears before the organ. The displacement stabs go to a decimated "
       "track. A 5-in-13 tuned metal voice runs over the sixteenths. The hook and guitar sag in "
       "pitch into the 2/4 jolt and snap back. The lift gets the dropped snare on the backbeat "
       "and a darker bass that slides into its octave pops. The return is the major light (G "
       "sharp in the hook, B and F sharp in the guitar), and major and minor alternate by "
       "two-bar phrase in the final. The bridge is a valley: one held root per bar, the kick "
       "out for its last two bars and the bass for its last, so bar 55 lands as a return. The "
       "lead runs over the hook in bars 57-62, the one place everything plays, ring-modulated "
       "the second time. Kick and kit transients are soft-clipped, and the hook and guitar come "
       "up. The limiter still takes 6.5 dB, beyond the 3 dB rule; Jonathan heard this render "
       "and rated it highly, so the rule yields to his ear here.",
}


def build_mix_v2(rev, g, n_bars, sec, length, dr, bass, hook, gtr, stab, lead, org, fog,
                 met=(), stab_dec=()):
    v3 = rev >= 3
    bass_patch = dict(PATCHES["bass"], sub=0.4)
    hook_patch = dict(PATCHES["hook"], cutoff=2400)
    gtr_eq = ("eq", {"bands": [["peak", 1500, 1.2, 3.0], ["peak", 3200, 1.0, 2.0]]})
    lead_inserts, hook_drag, gtr_drag = [], [], []
    if v3:
        # the hook is made audible through its sound before its fader; the bass gives up level
        bass_patch.update(sub=0.3, glide=0.07)
        hook_patch.update(oct=0.5, fenv=8000)
        # R18: the hook and guitar sag into the 2/4 jolt and snap back on bar 33's downbeat.
        # Varispeed reads at a cumulative rate, so the dragged bars go on their own tracks;
        # otherwise everything after the jolt would stay about a sixteenth late.
        t30, t32, t33 = g.bar(30), g.bar(32), g.bar(33)
        hook_drag = [n for n in hook if t30 <= n.t < t33]
        gtr_drag = [n for n in gtr if t30 <= n.t < t33]
        hook = [n for n in hook if not t30 <= n.t < t33]
        gtr = [n for n in gtr if not t30 <= n.t < t33]
        drag = ("varispeed", {"points": [(t30, 0), (t32, -120), (t33 - 0.01, -180), (t33, 0)]})
        # R14: the remembered melody comes back metallic, more so the second time
        lead_inserts = [("ringmod", {"freq": 311.0, "mix_points": [
            (g.bar(55), 0.0), (g.bar(57), 0.0), (g.bar(61), 0.35), (g.bar(63), 0.35)]})]
    hats = [n for n in dr if n.pitch in (42, 46)]
    kit = [n for n in dr if n.pitch not in (36, 42, 46)]
    # rev 3 (M2): the kick and kit transients are shaved before the master limiter sees them,
    # and the mid voices come up so loudness comes from the mids rather than the low end
    kick_ins = [("softclip", {"shave_db": 3.0})] if v3 else []
    kit_ins = [("softclip", {"shave_db": 2.0})] if v3 else []
    hook_db, gtr_db = (-4.5, -8.5) if v3 else (-5.5, -9.5)
    tracks = [
        Track("kick", "drums", PATCHES["drums"], [n for n in dr if n.pitch == 36],
              gain_db=-2.0 if v3 else -1.0, inserts=kick_ins),
        Track("kit", "drums", PATCHES["drums"], kit, gain_db=-2.5, pan=0.05, inserts=kit_ins),
        Track("hats", "drums", PATCHES["drums"], hats, gain_db=2.0, pan=0.22),
        Track("bass", "synth_bass", bass_patch, bass, gain_db=-5.5 if v3 else -4.0,
              duck=("kick", 3.0, 0.06), program=38),
        Track("hook", "blip", hook_patch, hook, gain_db=hook_db, pan=-0.35,
              sends={"slap": 0.5}, program=80),
        Track("guitar", "pluck", PATCHES["guitar"], gtr, gain_db=gtr_db, pan=0.45, program=28,
              inserts=[gtr_eq]),
        Track("stab", "fm", PATCHES["stab"], stab, gain_db=-8.0,
              stereo={"chorus": {"depth_ms": 2.0, "rate": 0.8, "mix": 0.5}}, sends={"slap": 0.3},
              program=61),
        Track("lead", "lead", PATCHES["lead"], lead, gain_db=-9.0, pan=0.1,
              sends={"echo": 0.35, "plate": 0.12}, program=81, inserts=lead_inserts),
        Track("organ", "organ", PATCHES["organ"], org, gain_db=-15.0,
              stereo={"chorus": {"depth_ms": 4.0, "rate": 0.35, "mix": 0.6}}, program=17,
              inserts=[("tape", {"drive": 1.2, "wow": 0.6, "flutter": 0.4, "hf": 7000})]),
        Track("fog", "pad", PATCHES["fog"], fog, gain_db=-16.0 if v3 else -12.0,
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.2, "mix": 0.7}},
              sends={"plate": 0.6}, program=91),
    ]
    if v3:
        # new tracks go last, so the earlier tracks keep their noise streams
        tracks += [
            Track("metal", "metal", {"decay": 0.25, "click": 0.3}, list(met), gain_db=-11.0,
                  pan=-0.4, program=14),
            Track("stab_dec", "fm", PATCHES["stab"], list(stab_dec), gain_db=-8.0,
                  inserts=[("decimate", {"hold": 3, "bits": 8})],
                  stereo={"chorus": {"depth_ms": 2.0, "rate": 0.8, "mix": 0.5}},
                  sends={"slap": 0.3}, program=61),
            Track("hook_drag", "blip", hook_patch, hook_drag, gain_db=hook_db, pan=-0.35,
                  sends={"slap": 0.5}, program=80, inserts=[drag]),
            Track("gtr_drag", "pluck", PATCHES["guitar"], gtr_drag, gain_db=gtr_db, pan=0.45,
                  program=28, inserts=[gtr_eq, drag]),
        ]
    buses = {
        "slap": {"type": "echo", "echo": {"time_s": g.steps(3), "feedback": 0.35, "repeats": 3,
                                          "tone": 4500, "spread": 0.9}, "return_db": -3},
        "echo": {"type": "echo", "echo": {"time_s": g.steps(6), "feedback": 0.45, "repeats": 5,
                                          "tone": 3000, "spread": 0.6, "wow": 1.0, "sat": 0.5},
                 "return_db": -5},
        "plate": {"type": "reverb", "ir": {"decay": 2.2, "predelay": 0.02, "tone": 6500},
                  "return_db": -6},
    }
    master = {"target_lufs": -14.0, "ceiling": -1.5, "tail": 2.5, "linear_phase_mono": True,
              "eq": [["hp", 28, 0.7], ["peak", 320, 1.0, -1.5], ["highshelf", 7500, 0.7, 1.5]],
              "comp": {"thresh_db": -16, "ratio": 1.6, "attack": 0.02, "release": 0.2},
              "tape": {"drive": 0.6, "wow": 0.06, "flutter": 0.08, "hiss_db": -80, "hf": 16000,
                       "bump_db": 0.8}}
    if v3:
        master.update(true_peak=True, ceiling=-2.0, linear_phase_mono=True)
    sections = [(name, g.bar(a), g.bar(b)) for name, (a, b) in sec.items()]
    return Song("Rivet Logic", "a_rivet_logic", BPM, length, tracks, buses, master, sections,
                SEED, rev, REV_NOTES.get(rev, ""),
                meta={"bar_starts": [g.bar(i) for i in range(n_bars + 1)],
                      "bass_syncopation": [syncopation(BASS_E1), syncopation(BASS_E2),
                                           syncopation(BASS_A)]})
