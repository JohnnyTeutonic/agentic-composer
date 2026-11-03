"""Low-level DSP: oscillators, envelopes, filters, strings, FM.

Everything is mono float64 at SR unless stated. Frequencies may be scalars or per-sample arrays.
"""
import numpy as np
from scipy import signal

SR = 44100


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def n_of(seconds):
    return max(1, int(round(seconds * SR)))


def _arr(x, n):
    return np.broadcast_to(np.asarray(x, dtype=float), (n,)).astype(float)


# ---------------------------------------------------------------- oscillators

def phase(f, n, phase0=0.0):
    dt = _arr(f, n) / SR
    ph = (phase0 + np.cumsum(dt) - dt) % 1.0
    return ph, dt


def _blep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt[m]
    out[m] = x * x + x + x + 1.0
    return out


def saw(f, n, phase0=0.0):
    t, dt = phase(f, n, phase0)
    return 2.0 * t - 1.0 - _blep(t, dt)


def pulse(f, n, width=0.5, phase0=0.0):
    t, dt = phase(f, n, phase0)
    w = np.clip(_arr(width, n), 0.02, 0.98)
    y = np.where(t < w, 1.0, -1.0)
    y += _blep(t, dt)
    y -= _blep((t - w) % 1.0, dt)
    return y - (2.0 * w - 1.0)


def sine(f, n, phase0=0.0):
    t, _ = phase(f, n, phase0)
    return np.sin(2.0 * np.pi * t)


def tri(f, n, phase0=0.0):
    t, _ = phase(f, n, phase0)
    return 4.0 * np.abs(t - 0.5) - 1.0


def fm(fc, n, ratio=1.0, index=1.0, ratio2=None, index2=0.0, phase0=0.0):
    """Two- or three-operator phase modulation. index/index2 may be envelopes."""
    tc, _ = phase(fc, n, phase0)
    tm, _ = phase(_arr(fc, n) * ratio, n)
    mod = np.sin(2 * np.pi * tm)
    if ratio2 is not None:
        tm2, _ = phase(_arr(fc, n) * ratio2, n)
        mod = np.sin(2 * np.pi * tm + _arr(index2, n) * np.sin(2 * np.pi * tm2))
    return np.sin(2 * np.pi * tc + _arr(index, n) * mod)


# ---------------------------------------------------------------- envelopes

def adsr(gate, a=0.005, d=0.1, s=0.7, r=0.1):
    """Gate length in seconds; returns envelope of gate + release."""
    ng, na, nr = n_of(gate), n_of(a), n_of(r)
    e = np.empty(ng)
    ia = min(na, ng)
    e[:ia] = np.linspace(0.0, 1.0, na, endpoint=False)[:ia]
    if ng > na:
        tt = np.arange(ng - na) / SR
        e[na:] = s + (1.0 - s) * np.exp(-tt * 4.6 / max(d, 1e-4))
    rel = e[-1] * np.exp(-np.arange(nr) / SR * 6.9 / max(r, 1e-4))
    return np.concatenate([e, rel])


def perc(decay, n=None, attack=0.001, curve=1.0):
    n = n if n is not None else n_of(decay * 1.2 + attack)
    t = np.arange(n) / SR
    e = np.exp(-t * 6.9 / max(decay, 1e-4)) ** curve
    na = n_of(attack)
    e[:na] *= np.linspace(0, 1, na)
    return e


def fade_tail(x, seconds=0.004):
    k = min(len(x), n_of(seconds))
    x = x.copy()
    x[-k:] *= np.linspace(1, 0, k)
    return x


# ---------------------------------------------------------------- filters

def biquad(kind, f0, q=0.707, gain_db=0.0):
    f0 = float(np.clip(f0, 10.0, 0.45 * SR))
    w0 = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * np.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw),
                 A * ((A + 1) - (A - 1) * cw - sq)]
            a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
                 (A + 1) + (A - 1) * cw - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw),
                 A * ((A + 1) + (A - 1) * cw - sq)]
            a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
                 (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    b, a = np.array(b), np.array(a)
    return b / a[0], a / a[0]


def filt(x, kind, f0, q=0.707, gain_db=0.0):
    b, a = biquad(kind, f0, q, gain_db)
    return signal.lfilter(b, a, x, axis=-1)


def filt_tv(x, kind, f0, q=0.707, block=64):
    """Time-varying biquad: coefficients recomputed every `block` samples."""
    n = x.shape[-1]
    f0 = _arr(f0, n)
    qq = _arr(q, n)
    y = np.empty_like(x)
    zi = np.zeros(2)
    for i in range(0, n, block):
        b, a = biquad(kind, f0[i], qq[i])
        y[i:i + block], zi = signal.lfilter(b, a, x[i:i + block], zi=zi)
    return y


def ladder(x, f0, res=0.3, drive=1.0):
    """24 dB/oct resonant low-pass (two cascaded biquads), mild input drive."""
    q1 = 0.55 + 6.0 * res ** 1.5
    y = np.tanh(drive * x) / max(np.tanh(drive), 1e-6)
    y = filt_tv(y, "lp", f0, q1)
    return filt_tv(y, "lp", f0, 0.6)


def butter(x, kind, f0, order=2):
    sos = signal.butter(order, f0, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


# ---------------------------------------------------------------- strings

def ks_string(freq, dur, t60=1.5, bright=0.5, pick=0.18, excite=None, rng=None):
    """Karplus-Strong plucked string, vectorised one period at a time.

    Loop: y[n] = x[n] + g * sum_j h_j y[n - N - j], with h = smoothing [s, 1-2s, s] convolved
    with a linear-interpolation pair for fractional tuning. bright in [0, 1]: 1 is brightest.
    """
    rng = rng or np.random.default_rng(0)
    P = SR / float(freq)
    s = 0.25 * (1.0 - np.clip(bright, 0, 1)) + 0.02
    N = int(np.floor(P - 1.0))
    frac = P - 1.0 - N
    h = np.convolve([s, 1 - 2 * s, s], [1 - frac, frac])
    g = 10 ** (-3.0 / (freq * max(t60, 1e-3)))
    n = n_of(dur)
    if excite is None:
        e = rng.uniform(-1, 1, N + 1)
        e = signal.lfilter([1 - 0.6 * (1 - bright)], [1, -0.6 * (1 - bright)], e)
        k = max(1, int(pick * N))
        e = e - np.roll(e, k)
        e -= e.mean()
    else:
        e = excite
    x = np.zeros(n)
    m = min(len(e), n)
    x[:m] = e[:m]
    pad = N + 4
    y = np.zeros(n + pad)
    for k in range(0, n, N):
        hi = min(k + N, n)
        L = hi - k
        acc = x[k:hi].copy()
        for j, hj in enumerate(h):
            lo = pad + k - N - j
            acc += g * hj * y[lo:lo + L]
        y[pad + k:pad + hi] = acc
    return y[pad:]


# ---------------------------------------------------------------- utilities

def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(d):
    return 10 ** (np.asarray(d, dtype=float) / 20)


def pan_stereo(x, pan):
    """Constant-power pan, pan in [-1, 1]; pan may be per-sample."""
    p = (np.asarray(pan, dtype=float) + 1) * np.pi / 4
    return np.stack([x * np.cos(p), x * np.sin(p)])


def place(buf, x, start):
    """Add x (mono or stereo) into buf at sample `start`, clipping to the buffer."""
    if start >= buf.shape[-1]:
        return
    if start < 0:
        x = x[..., -start:]
        start = 0
    end = min(buf.shape[-1], start + x.shape[-1])
    buf[..., start:end] += x[..., :end - start]
