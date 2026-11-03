"""Sampled instruments: recorded notes, repitched to the nearest sampled note.

The piano is the Salamander Grand Piano V3 (Alexander Holm, CC-BY 3.0; see SAMPLES.md and
samples/README.md). Its notes are sampled every minor third (A, C, D#, F#) at 16 velocities;
this project keeps layers 4, 8, 12 and 16. A note is repitched by at most 1.5 semitones.
Nothing here draws randomness, so renders stay deterministic.
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .dsp import SR, n_of, filt

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PIANO_DIR = SAMPLES / "salamander"
PIANO_LAYERS = (4, 8, 12, 16)
_NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9,
          "A#": 10, "B": 11}


def _read_mono(path):
    """16- or 24-bit PCM WAV (mono or stereo) to a float mono array."""
    raw = path.read_bytes()
    i = raw.index(b"fmt ") + 8
    ch, rate = int.from_bytes(raw[i + 2:i + 4], "little"), int.from_bytes(raw[i + 4:i + 8], "little")
    bits = int.from_bytes(raw[i + 14:i + 16], "little")
    j = raw.index(b"data")
    size = int.from_bytes(raw[j + 4:j + 8], "little")
    data = np.frombuffer(raw[j + 8:j + 8 + size], dtype=np.uint8)
    if bits == 16:
        x = data.view("<i2").astype(np.float64) / 32768.0
    elif bits == 24:
        b = data[: len(data) // 3 * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    else:
        raise ValueError(f"{path.name}: {bits}-bit WAV not supported")
    x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    if rate != SR:
        raise ValueError(f"{path.name}: {rate} Hz, engine runs at {SR} Hz")
    return x


@lru_cache(maxsize=None)
def _piano_index():
    """{layer: {midi: path}} for the kept piano samples."""
    if not PIANO_DIR.exists():
        raise FileNotFoundError(f"piano samples missing: run samples/fetch_salamander.py "
                                f"(expected {PIANO_DIR})")
    idx = {}
    for p in PIANO_DIR.glob("*.wav"):
        m = re.fullmatch(r"([A-G]#?)(\d)v(\d+)", p.stem)
        if m:
            midi = 12 * (int(m.group(2)) + 1) + _NAMES[m.group(1)]
            idx.setdefault(int(m.group(3)), {})[midi] = p
    return idx


@lru_cache(maxsize=256)
def _piano_note(layer, midi_src):
    x = _read_mono(_piano_index()[layer][midi_src])
    on = np.argmax(np.abs(x) > 1e-3)          # trim leading silence
    return x[on:]


@lru_cache(maxsize=512)
def _piano_pitched(layer, pitch):
    """The nearest sampled note in this layer, repitched to `pitch` (fractional MIDI allowed)."""
    notes = _piano_index()[layer]
    src = min(notes, key=lambda m: (abs(m - pitch), m))
    x = _piano_note(layer, src)
    ratio = 2 ** ((pitch - src) / 12)
    if abs(ratio - 1) < 1e-6:
        return x
    t = np.arange(0, len(x) - 1, ratio)
    return np.interp(t, np.arange(len(x)), x)


def piano(note, patch, rng):
    """Sampled grand piano.

    Patch keys: layer_bias (shifts velocity-layer choice, -1..1), release (s, damper time after
    the gate), max_len (s, cap on ringing), bright (dB high shelf at 3 kHz; house pianos are
    bright), body (dB peak at 250 Hz), tune (cents).
    """
    vel = float(np.clip(note.vel, 0.0, 1.0))
    pos = np.clip(vel + 0.25 * patch.get("layer_bias", 0.0), 0, 1) * (len(PIANO_LAYERS) - 1)
    layer = PIANO_LAYERS[int(round(pos))]
    pitch = round(note.pitch + patch.get("tune", 0.0) / 100.0, 3)
    x = _piano_pitched(layer, pitch)
    # the layer's recorded loudness carries most of the dynamics; vel fine-tunes within it
    x = x * (0.6 + 0.4 * vel)
    rel = patch.get("release", 0.15)
    gate = n_of(note.dur)
    n = min(len(x), gate + n_of(rel), n_of(patch.get("max_len", 8.0)))
    y = x[:n].copy()
    if n > gate:
        k = n - gate
        y[gate:] *= np.exp(-np.arange(k) / max(1, n_of(rel) / 5))
    if patch.get("bright", 0.0):
        y = filt(y, "highshelf", 3000.0, 0.707, patch["bright"])
    if patch.get("body", 0.0):
        y = filt(y, "peak", 250.0, 1.0, patch["body"])
    return y
