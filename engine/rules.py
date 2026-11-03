"""Compositional rules from research/production_notes.md, as functions.

Rhythm cells are lists of (step, length_in_steps, pitch_offset, velocity) on a 16-step bar.
"""
import numpy as np


class Grid:
    """Bar/step clock with fixed swing (R11). Bars may have non-default lengths in steps."""

    def __init__(self, bpm, swing=0.5, steps_per_beat=4, bar_steps=16, overrides=None, n_bars=400):
        self.bpm = bpm
        self.swing = swing
        self.step_s = 60.0 / bpm / steps_per_beat
        self.lengths = [overrides.get(b, bar_steps) if overrides else bar_steps
                        for b in range(n_bars)]
        self.starts = np.concatenate([[0.0], np.cumsum(self.lengths)]) * self.step_s

    def bar(self, b):
        return float(self.starts[b])

    def t(self, b, step):
        """Time of step (may be fractional) in bar b. Odd steps are delayed by the swing."""
        base = self.starts[b] + step * self.step_s
        if int(step) % 2 == 1 and float(step).is_integer():
            base += (self.swing - 0.5) * 2 * self.step_s
        return float(base)

    def steps(self, k):
        return k * self.step_s


def euclid(k, n, rot=0):
    """Bjorklund pattern: k onsets spread over n steps, rotated (R8)."""
    if k <= 0:
        return [0] * n
    pattern = [1 if (i * k) % n < k else 0 for i in range(n)]
    # the modular construction places onsets evenly; rotate so step 0 is an onset by default
    first = pattern.index(1)
    pattern = pattern[first:] + pattern[:first]
    rot %= n
    return pattern[-rot:] + pattern[:-rot] if rot else pattern


def syncopation(cell):
    """Fraction of onsets off the quarter-note beats, weighted toward the 'a' positions (R1)."""
    if not cell:
        return 0.0
    w = {0: 0.0, 2: 0.5, 1: 1.0, 3: 1.0}
    return float(np.mean([w[int(s) % 4] for s, *_ in cell]))


def rotate(cell, k, length=16):
    """Displace every onset by k steps around the bar (R5)."""
    return sorted([((s + k) % length, *rest) for s, *rest in cell])


def thin(cell, rng, keep_first=True):
    """Remove one onset, never the downbeat (R3)."""
    idx = [i for i, c in enumerate(cell) if not (keep_first and c[0] == 0)]
    if not idx:
        return cell
    drop = rng.choice(idx)
    return [c for i, c in enumerate(cell) if i != drop]


def nudge(cell, rng, length=16):
    """Move one non-downbeat onset by one step, if the target is free (R3)."""
    taken = {c[0] for c in cell}
    idx = [i for i, c in enumerate(cell) if c[0] != 0]
    rng.shuffle(idx)
    for i in idx:
        s = cell[i][0]
        for d in rng.permutation([-1, 1]):
            t = (s + d) % length
            if t not in taken and t != 0:
                out = list(cell)
                out[i] = (t, *cell[i][1:])
                return sorted(out)
    return cell


def clip(length_steps, gate):
    """Articulated length (R4)."""
    return length_steps * gate


def walk(pitch_set, start, n, rng, step_weights=(0.5, 0.3, 0.12, 0.08), no_repeat=True):
    """Weighted random walk over an ordered pitch set: mostly steps, sometimes leaps (R17, R23)."""
    ps = sorted(pitch_set)
    i = int(np.argmin([abs(p - start) for p in ps]))
    out = []
    for _ in range(n):
        size = rng.choice(len(step_weights), p=np.array(step_weights) / sum(step_weights))
        if size == 0 and no_repeat:
            size = 1
        direction = rng.choice([-1, 1])
        j = i + direction * size
        if j < 0 or j >= len(ps):
            j = i - direction * size
        i = int(np.clip(j, 0, len(ps) - 1))
        out.append(ps[i])
    return out


def poisson_times(start, end, mean_gap, rng, min_gap=0.0):
    """Event times with exponential gaps and a minimum spacing (R17)."""
    t, out = start, []
    while True:
        t += max(min_gap, rng.exponential(mean_gap))
        if t >= end:
            return out
        out.append(t)


def loop_starts(period, start, end, offset=0.0):
    """Start times of a loop of the given period (R16)."""
    out = []
    t = start + offset
    while t < end:
        out.append(t)
        t += period
    return out
