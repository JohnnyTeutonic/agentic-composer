"""Instrument voices. Each voice renders one Note with a patch (a plain dict) into a mono array.

A patch's keys are the synthesis parameters; patches are saved as JSON next to each render.
"""
import numpy as np

from . import dsp
from .dsp import SR, mtof, n_of, adsr, perc, filt, filt_tv, ladder
from .fx import varispeed, smooth_noise


def P(patch, key, default):
    return patch.get(key, default)


# ================================================================ pitched voices

def synth_bass(note, patch, rng):
    """Two oscillators and a sub through a resonant 24 dB low-pass with its own envelope."""
    f = mtof(note.pitch)
    gate = note.dur
    env = adsr(gate, P(patch, "a", 0.002), P(patch, "d", 0.18), P(patch, "s", 0.55),
               P(patch, "r", 0.03))
    n = len(env)
    glide = note.params.get("from_pitch")
    if glide is not None:
        k = n_of(P(patch, "glide", 0.04))
        fp = np.full(n, f)
        fp[:k] = mtof(np.linspace(glide, note.pitch, k))
    else:
        fp = f
    det = P(patch, "detune", 0.07)
    x = (P(patch, "saw", 0.7) * dsp.saw(fp * 2 ** (det / 12), n)
         + P(patch, "pulse", 0.5) * dsp.pulse(fp * 2 ** (-det / 12), n, P(patch, "pw", 0.32))
         + P(patch, "sub", 0.6) * dsp.sine(np.asarray(fp) / 2, n))
    fenv = perc(P(patch, "fdecay", 0.12), n)
    vel = note.vel
    cut = P(patch, "cutoff", 260) * (1 + P(patch, "keytrack", 0.3) * (f / 110 - 1))
    cut = cut + P(patch, "fenv", 2400) * fenv * (0.4 + 0.6 * vel)
    cut = cut * note.params.get("bright", 1.0)
    y = ladder(x * 0.5, cut, P(patch, "res", 0.35), P(patch, "drive", 1.6))
    return y * env


def blip(note, patch, rng):
    """Clipped square/pulse motif voice with a fast filter blip (R4)."""
    f = mtof(note.pitch)
    env = adsr(note.dur, 0.001, P(patch, "d", 0.06), P(patch, "s", 0.4), P(patch, "r", 0.02))
    n = len(env)
    pw = P(patch, "pw", 0.5) + P(patch, "pwm", 0.0) * np.sin(
        2 * np.pi * P(patch, "pwm_rate", 0.3) * (note.t + np.arange(n) / SR))
    x = dsp.pulse(f, n, pw) + P(patch, "oct", 0.0) * dsp.pulse(f * 2, n, 0.5)
    fenv = perc(P(patch, "fdecay", 0.05), n)
    cut = (P(patch, "cutoff", 1400) + P(patch, "fenv", 5000) * fenv * note.vel) \
        * note.params.get("bright", 1.0)
    y = filt_tv(x, "lp", cut, P(patch, "q", 2.0))
    return y * env * 0.4


def fm_voice(note, patch, rng):
    """FM voice: bells, glass, brass-like stabs. Index follows its own envelope."""
    f = mtof(note.pitch) * 2 ** (note.params.get("cents", 0) / 1200)
    a = note.params.get("a", P(patch, "a", 0.002))
    env = adsr(note.dur, a, P(patch, "d", 0.6), P(patch, "s", 0.0), P(patch, "r", 0.6))
    n = len(env)
    ienv = perc(P(patch, "idecay", 0.3), n)
    index = note.params.get("index", P(patch, "index", 3.0))
    idx = P(patch, "index_floor", 0.2) * index + index * ienv * (0.5 + 0.5 * note.vel)
    ratio = note.params.get("ratio", P(patch, "ratio", 3.5))
    vib = 1 + P(patch, "vib", 0.0) * np.sin(2 * np.pi * 5.1 * np.arange(n) / SR) \
        * np.clip(np.arange(n) / SR / 0.6, 0, 1)
    y = dsp.fm(f * vib, n, ratio, idx, P(patch, "ratio2", None), P(patch, "index2", 0.0))
    y += P(patch, "body", 0.0) * dsp.sine(f, n)
    if P(patch, "lp", None):
        y = filt(y, "lp", P(patch, "lp", 8000), 0.7)
    return y * env * 0.35


def pluck(note, patch, rng):
    """Karplus-Strong guitar or bass. `mute` shortens decay (palm mute / chicken scratch)."""
    f = mtof(note.pitch)
    mute = note.params.get("mute", P(patch, "mute", 0.0))
    t60 = P(patch, "t60", 2.0) * (1 - 0.92 * mute)
    bright = P(patch, "bright", 0.6) * (0.6 + 0.4 * note.vel) * (1 - 0.5 * mute)
    ring = note.dur + P(patch, "ring", 0.15)
    y = dsp.ks_string(f, ring + 0.05, t60, bright, P(patch, "pick", 0.18), rng=rng)
    if P(patch, "sub", 0.0):
        y += P(patch, "sub", 0.0) * dsp.sine(f, len(y)) * perc(t60 * 0.4, len(y))
    y = np.tanh(P(patch, "drive", 1.0) * y * 2.0) / 2.0 if P(patch, "drive", 0) else y
    env = adsr(note.dur, 0.001, 1.0, 1.0, P(patch, "release", 0.05))
    m = min(len(env), len(y))
    y = y[:m] * env[:m]
    if P(patch, "body_lp", None):
        y = filt(y, "lp", P(patch, "body_lp", 3000), 0.8)
    if P(patch, "hp", None):
        y = filt(y, "hp", P(patch, "hp", 80), 0.7)
    return y * 0.9


def pad(note, patch, rng):
    """Detuned saw/pulse ensemble with a slow filter drift (R13)."""
    f = mtof(note.pitch)
    env = adsr(note.dur, P(patch, "a", 1.2), P(patch, "d", 1.0), P(patch, "s", 0.8),
               P(patch, "r", 2.0))
    n = len(env)
    voices = P(patch, "voices", 4)
    spread = P(patch, "spread", 0.12)
    x = np.zeros(n)
    drift = smooth_noise(n, 0.3, rng) * P(patch, "drift_cents", 6) / 1200
    for k in range(voices):
        det = (k - (voices - 1) / 2) / max(1, voices - 1) * spread
        fk = f * 2 ** (det / 12 + drift * (1 if k % 2 else -1))
        osc = dsp.saw if k % 2 == 0 else (lambda ff, nn: dsp.pulse(ff, nn, 0.35))
        x += osc(fk, n) / voices
    move = smooth_noise(n, P(patch, "move_rate", 0.15), rng)
    cut = P(patch, "cutoff", 1200) * 2 ** (P(patch, "move_oct", 1.0) * move)
    y = filt_tv(x, "lp", cut, P(patch, "q", 0.9), block=256)
    if P(patch, "formant", None):
        y = vowel(y, P(patch, "formant", "o"), rng, morph=P(patch, "formant_morph", 0.0))
    return y * env * 0.5


VOWELS = {"a": (730, 1090, 2440), "o": (570, 840, 2410), "u": (300, 870, 2240),
          "e": (530, 1840, 2480), "i": (270, 2290, 3010)}


def vowel(x, v, rng, morph=0.0):
    """Parallel formant band-passes; `morph` drifts between this vowel and 'o'."""
    f1 = np.array(VOWELS[v], float)
    f2 = np.array(VOWELS["o"], float)
    n = len(x)
    curve = (smooth_noise(n, 0.1, rng) + 1) / 2 * morph if morph else 0.0
    y = np.zeros(n)
    for k, gain in enumerate((1.0, 0.6, 0.3)):
        fk = f1[k] + (f2[k] - f1[k]) * curve
        y += gain * filt_tv(x, "bp", fk, 6.0, block=256)
    return y * 2.0


def lead(note, patch, rng):
    """Sustaining lead with delayed vibrato and portamento; ebow-like with slow attack."""
    f = mtof(note.pitch)
    env = adsr(note.dur, P(patch, "a", 0.08), P(patch, "d", 0.3), P(patch, "s", 0.85),
               P(patch, "r", 0.4))
    n = len(env)
    t = np.arange(n) / SR
    fp = np.full(n, f)
    src = note.params.get("from_pitch")
    if src is not None:
        k = n_of(P(patch, "glide", 0.12))
        fp[:k] = mtof(np.linspace(src, note.pitch, k))
    vib_depth = P(patch, "vib", 0.25) * np.clip((t - P(patch, "vib_delay", 0.4)) / 0.8, 0, 1)
    fp = fp * 2 ** (vib_depth * np.sin(2 * np.pi * P(patch, "vib_rate", 5.3) * t) / 12)
    x = P(patch, "saw", 0.6) * dsp.saw(fp, n) + P(patch, "sq", 0.4) * dsp.pulse(fp * 1.003, n, 0.45)
    x += P(patch, "fmix", 0.0) * dsp.fm(fp, n, P(patch, "ratio", 2.0), P(patch, "index", 1.5))
    cut = P(patch, "cutoff", 2200) * (1 + 0.5 * smooth_noise(n, 0.4, rng) * P(patch, "move", 0.3))
    y = filt_tv(x, "lp", cut, P(patch, "q", 1.2), block=128)
    if P(patch, "drive", 0):
        y = np.tanh(y * P(patch, "drive", 1.0)) / np.tanh(P(patch, "drive", 1.0))
    return y * env * 0.35


def organ(note, patch, rng):
    """Additive drawbar-like tone with tape flutter: a soft, slightly unstable keyboard."""
    f = mtof(note.pitch)
    env = adsr(note.dur, P(patch, "a", 0.03), 0.2, 1.0, P(patch, "r", 0.25))
    n = len(env)
    fl = 1 + P(patch, "flutter", 0.002) * smooth_noise(n, 3.0, rng)
    y = np.zeros(n)
    for mult, g in P(patch, "bars", [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.15)]):
        y += g * dsp.sine(f * mult * fl, n)
    return y * env * 0.25


def metal(note, patch, rng):
    """Modal struck metal (bars, pipes, plates): inharmonic partials with separate decays."""
    f = mtof(note.pitch)
    ratios = P(patch, "ratios", [1.0, 2.756, 5.404, 8.933, 13.34])
    dec = note.params.get("decay", P(patch, "decay", 1.2))
    n = n_of(dec * 1.3 + 0.02)
    y = np.zeros(n)
    for k, r in enumerate(ratios):
        if f * r > 0.45 * SR:
            continue
        amp = (0.7 ** k) * rng.uniform(0.7, 1.0)
        y += amp * dsp.sine(f * r, n, rng.uniform()) * perc(dec / (1 + 0.8 * k), n)
    click = rng.standard_normal(n_of(0.004)) * np.linspace(1, 0, n_of(0.004))
    y[:len(click)] += P(patch, "click", 0.3) * filt(click, "hp", 2000)
    return y * 0.3 * note.vel


def breath(note, patch, rng):
    """Band-passed noise whose centre wanders: wind, breath, tape air (R13, R15)."""
    env = adsr(note.dur, P(patch, "a", 1.5), 0.5, 1.0, P(patch, "r", 2.0))
    n = len(env)
    centre = mtof(note.pitch) * 2 ** (P(patch, "wander_oct", 1.0) * smooth_noise(
        n, P(patch, "wander_rate", 0.2), rng))
    x = rng.standard_normal(n)
    y = filt_tv(x, "bp", centre, P(patch, "q", 4.0), block=256)
    return y * env * note.vel * 0.6


# ================================================================ drums

def drum_kick(vel, patch, rng):
    n = n_of(P(patch, "decay", 0.35) * 1.3)
    t = np.arange(n) / SR
    f = P(patch, "f_end", 46) + (P(patch, "f_start", 160) - P(patch, "f_end", 46)) * np.exp(
        -t / P(patch, "p_tau", 0.03))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * perc(P(patch, "decay", 0.35), n)
    click = filt(rng.standard_normal(n_of(0.003)), "bp", 3000, 1.0) * P(patch, "click", 0.3)
    y[:len(click)] += click
    return np.tanh(y * P(patch, "drive", 1.4)) * vel


def drum_snare(vel, patch, rng, drop=None):
    """Snare: two body modes plus band-passed noise. `drop` (semitones) pitches it down over
    its decay: the treated, dropped-pitch hit (R10)."""
    dec = P(patch, "decay", 0.18)
    n = n_of(dec * 1.6 + 0.05)
    body = (dsp.sine(P(patch, "f1", 185), n) + 0.6 * dsp.sine(P(patch, "f2", 330), n)) \
        * perc(P(patch, "body_decay", 0.07), n)
    nz = filt(rng.standard_normal(n), "bp", P(patch, "noise_f", 3500), 0.8)
    nz = filt(nz, "hp", 900) * perc(dec, n)
    y = P(patch, "body", 0.8) * body + P(patch, "noise", 0.9) * nz
    drop = P(patch, "drop", 0) if drop is None else drop
    if drop:
        ratio = 2 ** (-drop / 12 * np.clip(np.arange(n) / (dec * SR), 0, 1))
        y = varispeed(y, ratio)
    return np.tanh(y * 1.2) * vel


def drum_hat(vel, patch, rng, open_=False):
    dec = P(patch, "open_decay", 0.32) if open_ else P(patch, "decay", 0.045)
    n = n_of(dec * 1.3 + 0.01)
    freqs = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * P(patch, "tune", 1.6)
    x = sum(dsp.pulse(fq, n, 0.5) for fq in freqs) / 6
    x = 0.6 * x + 0.4 * rng.standard_normal(n)
    x = filt(filt(x, "bp", P(patch, "bp", 9500), 0.9), "hp", 6500)
    return x * perc(dec, n, curve=1.0) * vel * 0.8


def drum_clap(vel, patch, rng):
    n = n_of(0.3)
    e = np.zeros(n)
    for k, off in enumerate((0.0, 0.011, 0.021, 0.032)):
        i = n_of(off) if off else 0
        seg = perc(0.008 if k < 3 else P(patch, "tail", 0.12), n - i)
        e[i:] = np.maximum(e[i:], seg)
    x = filt(rng.standard_normal(n), "bp", P(patch, "f", 1300), 1.4)
    return x * e * vel * 1.4


def drum_tom(vel, patch, rng, pitch=100.0):
    """Electronic tom: sine with a long downward sweep (syndrum-like), touch of noise."""
    dec = P(patch, "decay", 0.4)
    n = n_of(dec * 1.3)
    t = np.arange(n) / SR
    f = pitch * (1 + P(patch, "sweep", 0.8) * np.exp(-t / P(patch, "sweep_tau", 0.08)))
    f = f * 2 ** (-P(patch, "fall", 5) / 12 * np.clip(t / dec, 0, 1))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * perc(dec, n)
    y += 0.15 * filt(rng.standard_normal(n), "bp", pitch * 4, 1.0) * perc(0.02, n)
    return np.tanh(1.3 * y) * vel


def drum_rim(vel, patch, rng):
    n = n_of(0.05)
    y = dsp.sine(P(patch, "f", 1750), n) * perc(0.012, n) + 0.5 * dsp.sine(
        P(patch, "f", 1750) * 0.46, n) * perc(0.02, n)
    y += 0.4 * filt(rng.standard_normal(n), "hp", 3000) * perc(0.004, n)
    return y * vel


def drum_block(vel, patch, rng, pitch=900.0):
    n = n_of(0.12)
    y = dsp.sine(pitch, n) * perc(0.05, n) + 0.3 * dsp.sine(pitch * 2.7, n) * perc(0.02, n)
    return y * vel


def drum_shaker(vel, patch, rng):
    n = n_of(0.09)
    env = np.minimum(np.linspace(0, 1, n) * 4, 1) * perc(0.05, n)
    y = filt(filt(rng.standard_normal(n), "bp", 7000, 1.2), "hp", 4500)
    return y * env * vel * 0.7


def drum_cowbell(vel, patch, rng):
    n = n_of(0.35)
    x = 0.5 * dsp.pulse(587, n) + 0.5 * dsp.pulse(845, n)
    x = filt(x, "bp", 2600, 1.6) * perc(0.25, n, curve=1.0)
    return x * vel * 0.6


def drum_noise(vel, patch, rng):
    """A filtered noise hit with a falling band-pass: a 'treated' accent."""
    n = n_of(P(patch, "decay", 0.5) * 1.2)
    f = 5000 * 2 ** (-3 * np.arange(n) / n)
    y = filt_tv(rng.standard_normal(n), "bp", f, 3.0) * perc(P(patch, "decay", 0.5), n)
    return y * vel


# GM-numbered kit so the drum MIDI opens in any sequencer with sensible sounds.
GM = {36: "kick", 38: "snare", 40: "snare_drop", 39: "clap", 42: "hat", 46: "hat_open",
      37: "rim", 56: "cowbell", 41: "tom_low", 45: "tom_mid", 48: "tom_high", 76: "block_hi",
      77: "block_lo", 70: "shaker", 49: "noise"}


def drum_kit(note, patch, rng):
    voice = GM.get(int(note.pitch), "rim")
    sub = patch.get(voice, {})
    v = note.vel
    if voice == "kick":
        return drum_kick(v, sub, rng)
    if voice == "snare":
        return drum_snare(v, sub, rng)
    if voice == "snare_drop":
        return drum_snare(v, sub, rng, drop=sub.get("drop", 7))
    if voice == "clap":
        return drum_clap(v, sub, rng)
    if voice in ("hat", "hat_open"):
        return drum_hat(v, sub, rng, open_=(voice == "hat_open"))
    if voice == "rim":
        return drum_rim(v, sub, rng)
    if voice == "cowbell":
        return drum_cowbell(v, sub, rng)
    if voice.startswith("tom"):
        base = {"tom_low": 82, "tom_mid": 118, "tom_high": 165}[voice]
        return drum_tom(v, sub, rng, note.params.get("tom_hz", base))
    if voice.startswith("block"):
        return drum_block(v, sub, rng, 1250 if voice == "block_hi" else 760)
    if voice == "shaker":
        return drum_shaker(v, sub, rng)
    if voice == "noise":
        return drum_noise(v, sub, rng)
    return drum_rim(v, sub, rng)


VOICES = {"synth_bass": synth_bass, "blip": blip, "fm": fm_voice, "pluck": pluck, "pad": pad,
          "lead": lead, "organ": organ, "metal": metal, "drums": drum_kit,
          "breath": breath}

from .sampler import piano  # noqa: E402  (sampled voice; needs samples/salamander/)
VOICES["piano"] = piano
