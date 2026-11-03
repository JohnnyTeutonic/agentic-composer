"""Effects. Stereo signals are shape (2, n); most effects accept mono or stereo."""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .dsp import SR, filt, butter, n_of, undb


def _lfo(n, rate, rng, phase=None):
    t = np.arange(n) / SR
    ph = rng.uniform(0, 2 * np.pi) if phase is None else phase
    return np.sin(2 * np.pi * rate * t + ph)


def smooth_noise(n, rate, rng):
    """Band-limited random curve in roughly [-1, 1], changing about `rate` times a second."""
    k = max(4, int(n / SR * rate) + 4)
    pts = rng.uniform(-1, 1, k)
    xs = np.linspace(0, n, k)
    from scipy.interpolate import CubicSpline
    return np.clip(CubicSpline(xs, pts)(np.arange(n)), -1.5, 1.5)


def varispeed(x, ratio):
    """Read x at a time-varying speed (ratio per sample, 1 = unchanged). Length preserved."""
    n = x.shape[-1]
    r = np.broadcast_to(np.asarray(ratio, float), (n,))
    pos = np.cumsum(r) - r[0]
    pos = np.clip(pos, 0, n - 1)
    idx = np.arange(n)
    if x.ndim == 1:
        return np.interp(pos, idx, x)
    return np.stack([np.interp(pos, idx, c) for c in x])


def mod_delay(x, delay_s, depth_s=0.0, rate=0.5, rng=None, mod=None):
    """Delay line with modulated read position (fractional, linear interpolation)."""
    rng = rng or np.random.default_rng(1)
    n = x.shape[-1]
    m = mod if mod is not None else _lfo(n, rate, rng)
    d = (delay_s + depth_s * m) * SR
    pos = np.arange(n) - d
    idx = np.arange(n)
    if x.ndim == 1:
        return np.interp(pos, idx, x, left=0.0)
    return np.stack([np.interp(pos, idx, c, left=0.0) for c in x])


def tape(x, drive=1.0, wow=0.0, flutter=0.0, hiss_db=-90.0, hf=13000.0, bump_db=1.5, rng=None):
    """Tape stage: head bump, asymmetric saturation, HF loss, wow and flutter, hiss."""
    rng = rng or np.random.default_rng(2)
    y = filt(x, "lowshelf", 90, 0.7, bump_db)
    if drive > 0:
        y = np.tanh(drive * (y + 0.04)) - np.tanh(drive * 0.04)
        y /= max(np.tanh(drive), 1e-6) if drive >= 1 else 1.0
    y = butter(y, "low", hf, 2)
    n = y.shape[-1]
    if wow or flutter:
        # wow: slow random drift; flutter: two fast sinusoids. Both move a delay read head,
        # so the pitch deviation follows the derivative of the delay curve.
        m = wow * smooth_noise(n, 0.7, rng) + flutter * 0.15 * (_lfo(n, 6.3, rng)
                                                              + 0.4 * _lfo(n, 11.1, rng))
        y = mod_delay(y, 0.010, 0.002, mod=m, rng=rng)
    if hiss_db > -89:
        h = rng.standard_normal(y.shape) * undb(hiss_db)
        h = butter(h, "high", 2500, 1)
        y = y + h
    return y


def echo(x, time_s, feedback=0.45, repeats=6, tone=3500.0, spread=0.0, wow=0.0, sat=0.0,
         rng=None):
    """Tape-style echo as a sum of repeats, each darker than the last. Returns wet only.

    spread > 0 alternates repeats left/right (ping-pong) when x is mono; output is stereo.
    """
    rng = rng or np.random.default_rng(3)
    mono = x if x.ndim == 1 else x.mean(axis=0)
    n = mono.shape[-1]
    out = np.zeros((2, n))
    cur = mono.copy()
    for k in range(1, repeats + 1):
        cur = butter(cur, "low", max(400.0, tone * (0.85 ** (k - 1))), 1)
        if sat:
            cur = np.tanh(cur * (1 + sat)) / (1 + sat) ** 0.5
        shift = int(round(time_s * k * SR))
        if shift >= n:
            break
        tap = np.zeros(n)
        tap[shift:] = cur[:n - shift]
        if wow:
            tap = mod_delay(tap, 0.004, 0.0015 * wow, rate=0.4 + 0.1 * k, rng=rng)
        g = feedback ** k
        side = (1 if k % 2 else -1) * spread
        out[0] += tap * g * np.sqrt(0.5 * (1 - side))
        out[1] += tap * g * np.sqrt(0.5 * (1 + side))
    return out


def make_ir(decay=2.0, predelay=0.015, tone=6000.0, early=0.3, size=1.0, rng=None):
    """Stereo reverb impulse response: early reflections plus decorrelated decaying noise."""
    rng = rng or np.random.default_rng(4)
    length = n_of(decay * 1.3 + predelay + 0.05)
    t = np.arange(length) / SR
    ir = np.zeros((2, length))
    pd = n_of(predelay)
    for ch in range(2):
        tail = rng.standard_normal(length) * np.exp(-t * 6.9 / decay)
        # high frequencies decay faster
        dark = butter(tail, "low", tone * 0.35, 1)
        mix = np.linspace(0, 1, length) ** 0.6
        tail = tail * (1 - mix) + dark * mix * 1.6
        tail = butter(tail, "low", tone, 2)
        tail[:pd] = 0
        fade_in = min(length - pd, n_of(0.02 * size))
        tail[pd:pd + fade_in] *= np.linspace(0, 1, fade_in)
        ir[ch] = tail
        for _ in range(int(8 * size)):
            tt = pd + int(rng.uniform(0.002, 0.04 * size) * SR)
            if tt < length:
                ir[ch, tt] += early * rng.uniform(-1, 1)
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + 1e-12
    return ir


def convolve_ir(x, ir):
    mono = x if x.ndim == 1 else x.mean(axis=0)
    n = mono.shape[-1]
    out = np.stack([signal.fftconvolve(mono, ir[c])[:n] for c in range(2)])
    if x.ndim == 2:
        # keep some of the source's own stereo image in the wet signal
        out = 0.7 * out + 0.3 * np.stack([signal.fftconvolve(x[c], ir[c])[:n] for c in range(2)])
    return out


def chorus(x, depth_ms=3.0, rate=0.6, mix=0.5, rng=None):
    """Mono in, stereo out: two modulated taps in quadrature."""
    rng = rng or np.random.default_rng(5)
    mono = x if x.ndim == 1 else x.mean(axis=0)
    n = mono.shape[-1]
    ph = rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / SR
    l = mod_delay(mono, 0.012, depth_ms / 1000, mod=np.sin(2 * np.pi * rate * t + ph))
    r = mod_delay(mono, 0.012, depth_ms / 1000, mod=np.cos(2 * np.pi * rate * t + ph))
    return np.stack([mono * (1 - mix) + l * mix, mono * (1 - mix) + r * mix])


def ringmod(x, freq, mix=1.0):
    n = x.shape[-1]
    c = np.sin(2 * np.pi * np.cumsum(np.broadcast_to(freq, (n,))) / SR)
    return x * (1 - mix) + x * c * mix


def decimate_hold(x, hold, bits=None):
    """Sample-and-hold decimation (hold samples) with optional bit reduction."""
    n = x.shape[-1]
    idx = (np.arange(n) // max(1, int(hold))) * max(1, int(hold))
    y = x[..., idx]
    if bits:
        q = 2 ** (bits - 1)
        y = np.round(y * q) / q
    return y


def envelope(x, attack=0.005, release=0.08):
    """Peak envelope follower (mono). Attack/release via max-filter then smoothing."""
    a = np.abs(x if x.ndim == 1 else np.max(np.abs(x), axis=0))
    w = max(1, n_of(attack))
    a = maximum_filter1d(a, size=w * 2 + 1)
    pole = np.exp(-1.0 / (release * SR))
    return signal.lfilter([1 - pole], [1, -pole], a)


def compress(x, thresh_db=-18.0, ratio=3.0, attack=0.005, release=0.1, makeup_db=0.0,
             key=None, knee_db=6.0):
    """Feed-forward compressor; `key` gives an external side-chain (ducking)."""
    env = envelope(x if key is None else key, attack, release)
    lvl = 20 * np.log10(np.maximum(env, 1e-9))
    over = lvl - thresh_db
    # soft knee
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = undb(-gr + makeup_db)
    return x * g


def limiter(x, ceiling_db=-1.2, lookahead=0.004, release=0.06, true_peak=False):
    """Look-ahead brick-wall limiter on a stereo signal.

    true_peak=False detects sample peaks (revision 1-2 behaviour). true_peak=True detects on a
    4x oversampled copy, so inter-sample overs are caught as well.
    """
    ceil = undb(ceiling_db)
    la = n_of(lookahead)
    if true_peak:
        up = signal.resample_poly(x, 4, 1, axis=-1)
        n = x.shape[-1]
        peak = np.max(np.abs(up[:, :n * 4]).reshape(2, n, 4), axis=(0, 2))
    else:
        peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    need = -maximum_filter1d(-need, size=2 * la + 1)   # min over the window
    pole = np.exp(-1.0 / (release * SR))
    # gain follows the minimum instantly and recovers with `release`
    g = np.empty_like(need)
    # vectorised recovery: smooth with one-pole on (1 - need) peak-held
    hold = maximum_filter1d(1 - need, size=la * 2 + 1)
    sm = signal.lfilter([1 - pole], [1, -pole], hold)
    g = 1 - np.maximum(hold, sm)
    g = uniform_filter1d(g, size=la)
    g = np.minimum(g, need)
    return x * g, g


def width(x, amount=1.0):
    """Mid/side width on a stereo signal (1 = unchanged)."""
    m = (x[0] + x[1]) / 2
    s = (x[0] - x[1]) / 2 * amount
    return np.stack([m + s, m - s])


def mono_below(x, freq=120.0, linear_phase=False):
    """Make everything below `freq` mono (M1).

    linear_phase=False is the revision-1 behaviour, kept for reproducibility: subtracting an IIR
    low-pass leaves a phase-shifted low residue in the high band, so lows stay partly stereo.
    linear_phase=True uses a zero-phase low-pass, so the split is exact.
    """
    if linear_phase:
        sos = signal.butter(4, freq, btype="low", fs=SR, output="sos")
        low = signal.sosfiltfilt(sos, x, axis=-1)
        lm = low.mean(axis=0, keepdims=True)
        return x - low + np.repeat(lm, 2, axis=0)
    low = butter(x, "low", freq, 4)
    high = x - low
    lm = low.mean(axis=0, keepdims=True)
    return high + np.repeat(lm, 2, axis=0)
