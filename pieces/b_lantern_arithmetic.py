"""B. Lantern Arithmetic: generative miniature, nominal 76 BPM, a drifting pedal under
incommensurate loops (R16), sparse events (R17) and composed treatments (R15).

The score is generated from a seed. The rules fix the materials; chance picks the instances.
"""
import numpy as np

from engine.rules import walk, poisson_times, loop_starts, euclid
from engine.song import Song, Track, Note
from pieces.common import nn

BPM = 76
SEED = 2207
LENGTH = 150.0

# Harmony: (start s, pedal, pitch-class set). The pedal moves rarely; loops follow by
# nearest-tone voice leading, so the harmony turns slowly underneath repeating material.
HARMONY = [
    (0.0, "Db2", {1, 5, 8, 0, 3}),          # Db maj9 colour
    (38.0, "Gb1", {6, 10, 1, 5, 8, 0}),     # Gb lydian: C is the raised fourth
    (72.0, "Bb1", {10, 1, 5, 8, 0, 3}),     # Bb minor 9 colour
    (104.0, "Ab1", {8, 0, 3, 6, 10, 1}),    # Ab mixolydian
    (130.0, "Db2", {1, 5, 8, 0, 3}),        # home, with C (maj7) foregrounded
]

# Loops: (name, period s, offset s, phrase as [(onset within loop, dur, pitch)])
LOOPS = [
    ("glass", 11.3, 1.0, [(0.0, 2.6, "F5"), (1.9, 3.0, "C5")]),
    ("swell", 17.9, 4.5, [(0.0, 7.0, "Ab4")]),
    ("bell", 23.3, 9.0, [(0.0, 4.0, "Eb4"), (0.7, 4.0, "Db5"), (2.2, 5.0, "Ab4")]),
    ("ebow", 29.1, 15.0, [(0.0, 6.0, "C4"), (5.0, 5.5, "Bb3")]),
]

PATCHES = {
    "drone": {"a": 5.0, "d": 2.0, "s": 0.9, "r": 6.0, "voices": 3, "spread": 0.08,
              "cutoff": 320, "move_oct": 0.8, "move_rate": 0.06, "q": 1.4, "drift_cents": 4},
    "glass": {"ratio": 3.0, "index": 1.2, "idecay": 0.6, "a": 0.01, "d": 2.5, "s": 0.0,
              "r": 1.5, "index_floor": 0.05, "body": 0.2},
    "swell": {"a": 2.8, "d": 2.0, "s": 0.8, "r": 3.0, "voices": 4, "spread": 0.1,
              "cutoff": 1400, "move_oct": 1.0, "move_rate": 0.2, "formant": "u",
              "formant_morph": 0.8},
    "bell": {"ratio": 3.5, "index": 2.5, "idecay": 0.9, "a": 0.002, "d": 3.0, "s": 0.0,
             "r": 2.0, "index_floor": 0.1},
    "ebow": {"saw": 0.4, "sq": 0.3, "fmix": 0.3, "ratio": 2.0, "index": 0.8, "a": 1.2,
             "s": 0.9, "r": 1.8, "cutoff": 1500, "q": 0.9, "vib": 0.1, "vib_delay": 1.5,
             "vib_rate": 4.6, "move": 0.5},
    "events": {"ratio": 1.4, "index": 2.0, "idecay": 0.5, "a": 0.003, "d": 2.2, "s": 0.0,
               "r": 1.6, "index_floor": 0.08},
    "air": {"a": 4.0, "r": 5.0, "q": 5.0, "wander_oct": 1.2, "wander_rate": 0.15},
    "pulse": {"block_lo": {}, "rim": {"f": 1500}},
}


def harmony_at(t):
    cur = HARMONY[0]
    for h in HARMONY:
        if t >= h[0]:
            cur = h
    return cur


def nearest_in(p, pcs):
    """Nearest pitch to p whose class is in pcs (ties go down: falling voice leading)."""
    for d in (0, -1, 1, -2, 2, -3, 3):
        if (p + d) % 12 in pcs:
            return p + d
    return p


# ==================================================================== revision 3 material
# One real departure in the middle: a chromatic mediant (A, the flat sixth of Db) and a
# dominant that borrows Cb from the minor. C is absent from 72 s to 130 s, so the home
# major seventh returns as an event (R22).
HARMONY_V3 = [
    (0.0, "Db2", {1, 5, 8, 0, 3}),          # Db maj9 colour
    (38.0, "Gb1", {6, 10, 1, 8, 0}),        # Gb lydian without F, so the glass must move
    (72.0, "A1", {9, 1, 4, 8, 11}),         # A maj9: bVI of Db; Db and Ab kept as C# and G#
    (104.0, "Ab1", {8, 11, 3, 6, 10, 1}),   # Ab with Cb borrowed from the minor
    (130.0, "Db2", {1, 5, 8, 0, 3}),        # home
]
HOME_V3 = 4
# accretion: the full four-loop texture first arrives in the Gb section
OFFSETS_V3 = {"glass": 1.0, "swell": 9.0, "bell": 21.0, "ebow": 40.0}
# loops leave before the floor; the glass stays to the end
STOPS_V3 = {"ebow": 127.0, "bell": 133.0, "swell": 130.0}
# drone notes (start, end): a hole after the A pedal, a late Ab, a short home pedal
DRONE_V3 = [(0.0, 41.0), (36.0, 73.0), (70.0, 88.0), (101.0, 131.0), (128.0, 134.0)]
# the motif: three rising steps in the pentatonic set and a falling sixth (R23)
MOTIF = [(0.0, "Eb5"), (0.39, "F5"), (0.92, "Ab5"), (1.84, "C5")]
MOTIF_DURS = [0.6, 0.8, 1.3, 2.6]
BOUNDS_V3 = [h[0] for h in HARMONY_V3[1:]]


def section_v3(t):
    i = 0
    for k, h in enumerate(HARMONY_V3):
        if t >= h[0]:
            i = k
    return i


def loop_sounds_v3(name, t):
    if name == "swell" and 70.0 <= t < 102.0:       # the v2 rest in the middle section
        return False
    return t < STOPS_V3.get(name, 1e9)


def realise_loops_v3(log, forced):
    """Re-voice the rev-1 loop decisions (mutation type, slot, shift, velocities) over the
    v3 harmony. Each note takes the harmony at its midpoint, so a long note that crosses a
    change is voiced for where it lands. At home every slot returns to the original phrase.
    `forced` holds (loop, section) pairs that must move to a tone the new set introduces."""
    out = {}
    for li, (name, period, _, phrase) in enumerate(LOOPS):
        jr = np.random.default_rng([SEED, 17, li])   # v3's own draws, one stream per loop
        orig = [nn(p) for _, _, p in phrase]
        pitches = list(orig)
        params = {}
        recs = []
        done = set()
        for k, start in enumerate(loop_starts(period, 0.0, LENGTH - 10, OFFSETS_V3[name])):
            d = log[name][k]
            if name == "ebow" and 94.0 <= start < 104.0:
                start = 104.0                         # the ebow re-enters on the return
            shift = d["shift"]
            secs = [section_v3(start + on + shift + 0.5 * dur) for on, dur, _ in phrase]
            for j in range(len(phrase)):
                pcs = HARMONY_V3[secs[j]][2]
                base = orig[j] if secs[j] == HOME_V3 else pitches[j]
                pitches[j] = nearest_in(base, pcs)
                s = secs[j]
                if ((name, s) in forced and (name, s) not in done
                        and loop_sounds_v3(name, start + phrase[j][0] + shift)):
                    new = pcs - HARMONY_V3[s - 1][2]
                    cand = [q for q in range(pitches[j] - 4, pitches[j] + 5)
                            if q % 12 in new and q % 12 != pitches[j] % 12]
                    if cand:
                        pitches[j] = min(cand, key=lambda q: (abs(q - pitches[j]), q))
                        done.add((name, s))
            if d["choice"] == "pitch":
                j = d["j"]
                pcs = HARMONY_V3[secs[j]][2]
                cand = [q for q in range(pitches[j] - 4, pitches[j] + 5)
                        if q % 12 in pcs and q != pitches[j]]
                pitches[j] = int(min(cand, key=lambda q: abs(q - pitches[j])
                                     + jr.uniform(0, 2)))
            elif d["choice"] == "timbre":
                if name == "bell":                    # the metal voice mutates its decay
                    params = {"decay": float(jr.uniform(2.4, 5.0))}
                else:
                    params = dict(d["params"])
            for j, ((on, dur, _), p) in enumerate(zip(phrase, pitches)):
                t = start + on + shift
                recs.append({"t": t, "dur": dur, "pitch": int(p), "vel": d["vels"][j],
                             "params": dict(params), "sec": secs[j], "slot": j,
                             "on": loop_sounds_v3(name, t)})
        out[name] = recs
    return out


def voices_changed_v3(real, s):
    """How many sounding loop voices change pitch class across the change into section s,
    and the unchanged ones ordered by when they first sound in s."""
    n, unchanged = 0, []
    for name, recs in real.items():
        before, after, first = {}, {}, None
        for r in sorted((r for r in recs if r["on"]), key=lambda r: r["t"]):
            if r["sec"] < s:
                before[r["slot"]] = r["pitch"] % 12
            elif r["sec"] == s and r["slot"] not in after:
                after[r["slot"]] = r["pitch"] % 12
                first = r["t"] if first is None else first
        common = set(before) & set(after)
        if not common:
            continue
        if any(before[j] != after[j] for j in common):
            n += 1
        else:
            unchanged.append((first, name))
    return n, [nm for _, nm in sorted(unchanged)]


def motif_notes(t0, k, pcs, invert=False, octave=0, vel=0.7, last_dur=None):
    base = nn("Eb5")
    out = []
    for j, ((on, name), dur) in enumerate(zip(MOTIF[:k], MOTIF_DURS[:k])):
        p = nn(name)
        if invert:
            p = base - (p - base)
        p = nearest_in(p + 12 * octave, pcs)
        if j == k - 1:
            dur = last_dur or max(dur, 1.6)
        out.append(Note(t0 + on, dur, p, vel))
    return out


def dips_v3():
    """Tape-speed dips at irregular, seeded times; at least two away from section changes."""
    tr = np.random.default_rng([SEED, 13])
    for _ in range(50):
        times = poisson_times(18.0, 135.0, 24.0, tr, min_gap=12.0)
        away = [t for t in times if min(abs(t - b) for b in BOUNDS_V3) > 4.0]
        if len(away) >= 2:
            break
    pts = [(0.0, 0.0)]
    for t in times:
        fall = float(tr.uniform(0.6, 1.4))
        depth = float(tr.uniform(-60.0, -15.0))
        rec = float(tr.uniform(1.0, 3.0))
        pts += [(t - fall, 0.0), (t, depth), (t + rec, 0.0)]
    pts.append((LENGTH + 10, 0.0))
    return pts, times


def build(rev=1):
    # the realisation is fixed to revision 1 so later revisions change only what they say
    rng = np.random.default_rng([SEED, 1])
    tracks_notes = {name: [] for name in ("drone", "glass", "swell", "bell", "ebow", "events",
                                          "air", "pulse")}

    # ---------------------------------------------------------------- drone (pedal)
    for i, (t0, ped, _) in enumerate(HARMONY):
        t1 = HARMONY[i + 1][0] if i + 1 < len(HARMONY) else LENGTH - 6
        for p in (nn(ped), nn(ped) + 7):            # root and fifth, no third (R22)
            tracks_notes["drone"].append(Note(max(0.0, t0 - 2.0), t1 - t0 + 3.0, p,
                                              0.8 if p == nn(ped) else 0.45))

    # ---------------------------------------------------------------- loops (R16, R18)
    end = LENGTH - 10
    loop_log = {}       # the rev-1 decisions per repeat, re-voiced by revision 3
    for name, period, offset, phrase in LOOPS:
        pitches = [nn(p) for _, _, p in phrase]
        params = {}
        loop_log[name] = []
        for k, start in enumerate(loop_starts(period, 0.0, end, offset)):
            pcs = harmony_at(start)[2]
            pitches = [nearest_in(p, pcs) for p in pitches]
            # mutation budget: at most one change per repeat (pitch, timbre or placement)
            shift = 0.0
            rec = {"choice": "none", "j": None}
            if k > 0:
                choice = rng.choice(["none", "pitch", "timbre", "place"], p=[0.4, 0.25, 0.2, 0.15])
                rec["choice"] = str(choice)
                if choice == "pitch":
                    j = rng.integers(len(pitches))
                    rec["j"] = int(j)
                    cand = [q for q in range(pitches[j] - 4, pitches[j] + 5)
                            if q % 12 in pcs and q != pitches[j]]
                    pitches[j] = int(min(cand, key=lambda q: abs(q - pitches[j])
                                         + rng.uniform(0, 2)))
                elif choice == "timbre":
                    params = {"ratio": float(rng.choice([1.0, 1.41, 2.0, 2.76, 3.5])),
                              "index": float(rng.uniform(0.6, 3.0))}
                elif choice == "place":
                    shift = float(rng.uniform(-0.8, 0.8))
            rec.update(shift=shift, params=dict(params), vels=[])
            for (on, dur, _), p in zip(phrase, pitches):
                vel = float(rng.uniform(0.5, 0.8))
                rec["vels"].append(vel)
                tracks_notes[name].append(Note(start + on + shift, dur, p, vel, dict(params)))
            loop_log[name].append(rec)

    # ---------------------------------------------------------------- sparse events (R17)
    ev_times = poisson_times(6.0, LENGTH - 12, 5.5, rng, min_gap=2.5)
    last = nn("Ab5")
    ev_log = []         # (time, notes) per rev-1 event, for revision 3
    for t in ev_times:
        pcs = harmony_at(t)[2]
        pool = [q for q in range(nn("Ab4"), nn("F6") + 1) if q % 12 in pcs]
        k = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
        seq = walk(pool, last, k, rng)
        last = seq[-1]
        params = {"ratio": float(rng.choice([1.0, 1.41, 2.0, 2.76, 3.5])),
                  "index": float(rng.uniform(0.8, 3.2))}
        ev_notes = []
        for j, p in enumerate(seq):
            ev_notes.append(Note(t + j * float(rng.choice([0.26, 0.39, 0.53])),
                                 1.2, p, float(rng.uniform(0.35, 0.75)), dict(params)))
        tracks_notes["events"].extend(ev_notes)
        ev_log.append((t, ev_notes))

    # ---------------------------------------------------------------- treatments (R15)
    # reversed swells landing on harmony changes
    for t0, ped, pcs in HARMONY[1:]:
        p = nearest_in(nn("Db5"), pcs)
        tracks_notes["glass"].append(Note(t0 - 3.0, 3.0, p, 0.7, {"reverse": True}))
    # air: wandering band-passed noise, two long phrases
    tracks_notes["air"].append(Note(20.0, 40.0, nn("C7"), 0.35))
    tracks_notes["air"].append(Note(95.0, 40.0, nn("G6"), 0.3))
    # a faint clock only in the middle (euclid 5 in 13 eighths)
    # (revision 3: the clock counts 3, 5 then 7 in 13; the pulse is the last consumer of rng)
    step = 60 / BPM / 2
    clock = ([(40.0, 72.0, 3), (72.0, 104.0, 5), (104.0, 128.0, 7)] if rev >= 3
             else [(74.0, 118.0, 5)])
    for c0, c1, onsets in clock:
        pat = euclid(onsets, 13)
        t, i = c0, 0
        while t < c1:
            if pat[i % 13]:
                tracks_notes["pulse"].append(Note(t, 0.05, 77 if i % 26 < 13 else 37,
                                                  float(rng.uniform(0.25, 0.4))))
            t += step
            i += 1

    # tape-speed dips (cents) applied to the loop group
    dips = [(0.0, 0.0), (58.0, 0.0), (59.5, -45.0), (61.5, 0.0), (115.0, 0.0),
            (116.0, -30.0), (118.5, 10.0), (120.0, 0.0), (LENGTH + 10, 0.0)]
    vs = ("varispeed", {"points": dips})
    ring = ("ringmod", {"freq": 311.0, "mix_points": [(0, 0), (76, 0), (80, 0.35), (96, 0.35),
                                                      (102, 0), (LENGTH + 10, 0)]})

    v2 = rev >= 2
    gains = {"drone": -7.0, "glass": -12.0, "swell": -12.0, "bell": -14.0, "ebow": -15.0,
             "events": -13.0, "air": -22.0}
    patches = {k: dict(v) for k, v in PATCHES.items()}
    if v2:
        tn = tracks_notes
        # an arc: the Bb section thins (swell loop rests), Ab blooms (denser events)
        tn["swell"] = [n for n in tn["swell"] if not 70.0 <= n.t < 102.0]
        vr = np.random.default_rng([SEED, 7])
        for t in poisson_times(104.0, 127.0, 3.0, vr, min_gap=1.6):
            pcs = harmony_at(t)[2]
            pool = [q for q in range(nn("Eb5"), nn("Ab6") + 1) if q % 12 in pcs]
            p = int(vr.choice(pool))
            tn["events"].append(Note(t, 1.0, p, float(vr.uniform(0.3, 0.6)),
                                     {"ratio": 2.0, "index": float(vr.uniform(0.6, 1.6))}))
        # home: the loops leave one at a time (R19), the glass stays
        for name, stop in (("ebow", 128.0), ("bell", 133.0), ("swell", 137.0)):
            tn[name] = [n for n in tn[name] if n.t < stop]
        gains.update(drone=-11.0, glass=-9.5, bell=-11.0, events=-10.5, air=-19.0)
        patches["drone"]["cutoff"] = 260
        patches["glass"].update(index=1.6, body=0.1)

    v3 = rev >= 3
    forced = set()
    if v3:
        tn = tracks_notes
        # -------- drone: root and fifth, a hole before the return, a short home pedal
        tn["drone"] = []
        for (_, ped, _), (a, b) in zip(HARMONY_V3, DRONE_V3):
            for p in (nn(ped), nn(ped) + 7):
                tn["drone"].append(Note(a, b - a, p, 0.8 if p == nn(ped) else 0.3))
        # -------- loops: rev-1 decisions re-voiced; at least two voices move at each change
        # (at least two must move; three where three can, so no loop parks on a common tone)
        real = realise_loops_v3(loop_log, forced)
        for s in range(1, len(HARMONY_V3)):
            n_changed, unchanged = voices_changed_v3(real, s)
            if n_changed < 3 and unchanged:
                forced.update((nm, s) for nm in unchanged[:3 - n_changed])
                real = realise_loops_v3(loop_log, forced)
        for name in ("glass", "swell", "bell", "ebow"):
            tn[name] = [Note(r["t"], r["dur"], r["pitch"], r["vel"], r["params"])
                        for r in real[name] if r["on"]]
        # the texture breaks once: the glass through the static stretch goes through a crusher
        tn["glass_crushed"] = [n for n in tn["glass"] if 56.0 <= n.t < 70.0]
        tn["glass"] = [n for n in tn["glass"] if not 56.0 <= n.t < 70.0]
        # reversed glass: two announce the new tone at a change, two arrive mid-section
        for t0, p in ((50.0, "Bb4"), (72.0, "B4"), (117.0, "B4"), (130.0, "C5")):
            tn["glass"].append(Note(t0 - 3.0, 3.0, nn(p), 0.7, {"reverse": True}))
        # three ringmodded metal strikes inside the hole, and the second breath enters early
        tn["bell"] += [Note(96.8, 2.0, nn("E5"), 0.85), Note(97.5, 2.0, nn("B4"), 0.7),
                       Note(100.1, 2.0, nn("C#5"), 0.6)]
        tn["air"] = [tn["air"][0], Note(92.0, 43.0, nn("G6"), 0.3)]
        # -------- the motif: stated once, then chance picks its fragments (R17, R23)
        mr = np.random.default_rng([SEED, 11])
        tn["events"], tn["motif"] = [], []
        prefix_t = [106.0, 112.0, 118.0, 124.0]
        tn["motif"] += motif_notes(24.0, 4, HARMONY_V3[0][2], vel=0.75)
        # the A section hears the motif upside down, in full, before the hole
        tn["motif"] += motif_notes(83.0, 4, HARMONY_V3[2][2], invert=True, vel=0.7)
        for t, ev_notes in ev_log:
            # the walks keep their rev-1 rhythm and timbre, re-pitched into the v3 harmony
            ev_notes = [Note(n.t, n.dur, nearest_in(n.pitch, HARMONY_V3[section_v3(n.t)][2]),
                             n.vel, dict(n.params)) for n in ev_notes]
            u = float(mr.random())
            k = int(mr.choice([1, 2, 3, 4], p=[0.35, 0.3, 0.2, 0.15]))
            vel = float(mr.uniform(0.45, 0.7))
            s = section_v3(t)
            if 22.0 <= t < 27.5 or t >= 129.0:
                continue                      # room for the statement; nothing after home
            if s == 3:                        # the Ab section belongs to the prefix sequence
                if u < 0.6 or min(abs(t - p) for p in prefix_t) < 2.5:
                    continue
                tn["events"] += ev_notes
            elif t > 27.5 and u < 0.6:
                tn["motif"] += motif_notes(t, k, HARMONY_V3[s][2], invert=(s == 2), vel=vel)
            else:
                tn["events"] += ev_notes
        # the Ab section builds an expectation with growing prefixes ...
        for k, (t, vel) in enumerate(zip(prefix_t, (0.5, 0.58, 0.66, 0.75)), start=1):
            tn["motif"] += motif_notes(t, k, HARMONY_V3[3][2], vel=vel)
        # ... answered at home, an octave lower, landing on the returning C
        tn["motif"] += motif_notes(131.0, 4, HARMONY_V3[4][2], octave=-1, vel=0.8,
                                   last_dur=4.0)
        # the downbeat of the return: one low struck Ab
        tn["strike"] = [Note(104.0, 4.0, nn("Ab1"), 0.9), Note(104.0, 4.0, nn("Ab2"), 0.55)]
        # -------- treatments as irregular events (R14, R15)
        dip_pts, dip_times = dips_v3()
        vs = ("varispeed", {"points": dip_pts})
        ring = ("ringmod", {"freq": 311.0, "mix_points": [(0, 0), (86, 0), (90, 0.5), (102, 0.5),
                                                          (105, 0), (LENGTH + 10, 0)]})
        ring_ebow = ("ringmod", {"freq": 233.0, "mix_points": [(0, 0), (41, 0), (43.5, 0.4),
                                                               (50, 0.4), (53, 0),
                                                               (LENGTH + 10, 0)]})
        # -------- sound and mix
        gains.update(drone=-15.0, glass=-8.0, bell=-13.0, events=-9.0, air=-17.0, motif=-11.5,
                     strike=-10.0)
        gains["glass_crushed"] = gains["glass"]
        patches["drone"]["cutoff"] = 1100
        patches["glass"].update(index=2.2, idecay=1.0)
        patches["bell"] = {"ratios": [1.0, 2.32, 4.25, 6.63, 9.4], "decay": 3.5, "click": 0.12}
        patches["motif"] = {"t60": 3.5, "bright": 0.4, "body_lp": 2500, "pick": 0.22}
        patches["strike"] = {"t60": 4.0, "bright": 0.45, "body_lp": 1800}

    drone_ins = ([("lp_auto", {"points": [(0, 260), (104, 260), (118, 1400), (128, 600),
                                          (138, 260)], "q": 1.1})] if v3 else [])
    ebow_ins = [vs] + ([ring_ebow] if v3 else []) + [
        ("tape", {"drive": 1.5, "wow": 0.8, "flutter": 0.3, "hf": 6000})]
    tracks = [
        Track("drone", "pad", patches["drone"], tracks_notes["drone"], gain_db=gains["drone"],
              stereo={"chorus": {"depth_ms": 3.0, "rate": 0.1, "mix": 0.5}}, inserts=drone_ins,
              sends={"hall": 0.08 if v3 else 0.3}, program=89),
        Track("glass", "fm", patches["glass"], tracks_notes["glass"], gain_db=gains["glass"], pan=-0.45,
              inserts=[vs], sends={"hall": 0.5, "echo": 0.3}, program=14),
        Track("swell", "pad", patches["swell"], tracks_notes["swell"], gain_db=gains["swell"],
              stereo={"chorus": {"depth_ms": 5.0, "rate": 0.15, "mix": 0.7}},
              inserts=[vs], sends={"hall": 0.5}, program=91),
        Track("bell", "metal" if v3 else "fm", patches["bell"], tracks_notes["bell"],
              gain_db=gains["bell"], pan=0.4, inserts=[vs, ring],
              sends={"hall": 0.25} if v3 else {"hall": 0.55, "echo": 0.25}, program=10),
        Track("ebow", "lead", patches["ebow"], tracks_notes["ebow"], gain_db=gains["ebow"], pan=0.15,
              inserts=ebow_ins, sends={"hall": 0.45, "echo": 0.4}, program=82),
        Track("events", "fm", patches["events"], tracks_notes["events"], gain_db=gains["events"], pan=-0.1,
              stereo={"haas": 9}, sends={"hall": 0.12 if v3 else 0.4, "echo": 0.5}, program=8),
        Track("air", "breath", patches["air"], tracks_notes["air"], gain_db=gains["air"],
              stereo={"chorus": {"depth_ms": 8.0, "rate": 0.07, "mix": 0.8}},
              sends={"hall": 0.4}, program=122),
        Track("pulse", "drums", PATCHES["pulse"], tracks_notes["pulse"],
              gain_db=-15.0 if v3 else -20.0, pan=0.25, sends={"hall": 0.35}),
    ]
    if v3:
        # appended after the rev-1 tracks so their per-track noise draws are unchanged
        tracks += [
            Track("glass_crushed", "fm", patches["glass"], tracks_notes["glass_crushed"],
                  gain_db=gains["glass_crushed"], pan=-0.45,
                  inserts=[("decimate", {"hold": 6, "bits": 9}), vs],
                  sends={"hall": 0.5, "echo": 0.3}, program=14),
            Track("motif", "pluck", patches["motif"], tracks_notes["motif"],
                  gain_db=gains["motif"], pan=0.3, inserts=[("softclip", {"shave_db": 2.0})],
                  sends={"hall": 0.2, "echo": 0.5}, program=25),
            Track("strike", "pluck", patches["strike"], tracks_notes["strike"],
                  gain_db=gains["strike"], inserts=[("softclip", {"shave_db": 2.0})], program=33),
        ]
    buses = {
        "hall": {"type": "reverb", "ir": {"decay": 6.5, "predelay": 0.04, "tone": 5000,
                                          "size": 2.0}, "return_db": -3},
        "echo": {"type": "echo", "echo": {"time_s": 60 / BPM * 1.5, "feedback": 0.55,
                                          "repeats": 7, "tone": 2800, "spread": 0.7,
                                          "wow": 1.5, "sat": 0.3}, "return_db": -6},
    }
    if v3:
        # the hall return is high-passed above the pedal's second harmonic
        buses["hall"]["inserts"] = [("hp", {"f": 220, "order": 2})]
    master = {"target_lufs": -17.0, "ceiling": -1.5, "tail": 6.0,
              "eq": [["hp", 30, 0.7]] + ([["highshelf", 6000, 0.7, 2.0]] if v2 and not v3 else []),
              "linear_phase_mono": v2,
              "tape": {"drive": 0.5, "wow": 0.15, "flutter": 0.08, "hiss_db": -74, "hf": 12500,
                       "bump_db": 1.5},
              "fade_out": [LENGTH - 2, LENGTH + 5.5]}
    if v3:
        master["true_peak"] = True
    sections = [("Db", 0, 38), ("Gb lydian", 38, 72), ("Bb", 72, 104), ("Ab", 104, 130),
                ("Db home", 130, LENGTH)]
    if v3:
        sections = [("Db", 0, 38), ("Gb lydian", 38, 72), ("A (bVI), hole", 72, 104),
                    ("Ab, Cb borrowed", 104, 130), ("Db home", 130, LENGTH)]
    notes = {1: "First complete generative render.",
             2: "Drone down 4 dB and darker; glass, bell and events up 2.5-3 dB, glass brighter; "
                "presence shelf; lows truly mono. Form arc: the swell loop rests through the Bb "
                "section, extra high events thicken the Ab section, and at home the loops leave "
                "one at a time (ebow, bell, swell) until glass and drone remain.",
             3: "The middle of the piece now departs: an A major ninth section, the flat sixth "
                "of D-flat, then an A-flat section that borrows C-flat from the minor, so C is "
                "absent from 72 to 130 seconds and returns at home as an event. Loop notes take "
                "the harmony at their midpoint and return to the original phrase at home, and "
                "at every pedal change at least two loop voices change pitch class, three where "
                "three are sounding. The loops accrete (swell 9 s, bell 21 s, ebow 40 s), the "
                "drone leaves a hole from about 94 to 104 seconds that holds only ringmodded "
                "metal strikes, breath and the clock, a low struck A-flat marks the return, the "
                "drone filter "
                "blooms through the A-flat section, and the floor leaves at 134 seconds so the "
                "glass ends alone. A four-note motif is stated at 24 seconds on a plucked "
                "string; chance then picks its fragments, inverted in the A section, the A-flat "
                "section grows it one note at a time, and an octave-lower statement at 131 "
                "seconds answers it. The Euclidean clock counts 3, 5 and 7 in 13 and is louder. "
                "The bell is now struck metal in a closer space, tape dips fall at seeded "
                "irregular times, a ringmod window colours the ebow, the glass is crushed once "
                "between 56 and 70 seconds, and two reversed swells moved mid-section. The hall "
                "return is high-passed at 220 Hz, the drone is 4 dB lower and sends far less to "
                "it, and the master presence shelf is gone in favour of brighter sources."}
    return Song("Lantern Arithmetic", "b_lantern_arithmetic", BPM, LENGTH, tracks, buses, master,
                sections, SEED, rev, notes.get(rev, ""), meta={"bar_starts": None})
