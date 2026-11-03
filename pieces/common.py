"""Helpers shared by the pieces: turning rhythm cells into notes."""
from engine.song import Note

NAMES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
         "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def nn(name):
    """'E5' -> 76, 'F#1' -> 30."""
    pc = name[:-1]
    return 12 * (int(name[-1]) + 1) + NAMES[pc]


def add_cell(notes, grid, bar, cell, root, gate=1.0, vel=1.0, jitter=0.0, rng=None,
             params=None):
    """cell: [(step, length_steps, semitone_offset, velocity)]."""
    for step, length, off, v in cell:
        t = grid.t(bar, step)
        if jitter and rng is not None:
            t += rng.normal(0, jitter)
        notes.append(Note(max(0.0, t), grid.steps(length) * gate, root + off, v * vel,
                          dict(params or {})))


def add_hits(notes, grid, bar, steps, gm, vel=0.8, jitter=0.0, rng=None, params=None):
    """steps: iterable of step numbers or (step, velocity) pairs."""
    for s in steps:
        step, v = (s, vel) if not isinstance(s, tuple) else s
        t = grid.t(bar, step)
        if jitter and rng is not None:
            t += rng.normal(0, jitter)
        notes.append(Note(max(0.0, t), grid.steps(1) * 0.5, gm, v, dict(params or {})))


def add_line(notes, grid, bar0, line, gate=0.95, vel=0.8, params=None):
    """line: [(bar_offset, step, length_steps, 'E5' or midi, velocity optional)]."""
    for item in line:
        b, step, length, p = item[:4]
        v = item[4] if len(item) > 4 else vel
        pitch = nn(p) if isinstance(p, str) else p
        notes.append(Note(grid.t(bar0 + b, step), grid.steps(length) * gate, pitch, v,
                          dict(params or {})))


def euclid_steps(pattern):
    return [i for i, x in enumerate(pattern) if x]
