"""Objective analysis of a rendered master, and a diagnostic figure.

These are measurements, not listening. Musical judgements in the diary are drawn from them plus
the score data.
"""
import re
import subprocess

import numpy as np
from scipy import signal

from .dsp import SR, butter
from .song import lufs, k_weight

BANDS = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 1000), ("mid", 1000, 4000),
         ("presence", 4000, 10000), ("air", 10000, 20000)]


def ffmpeg_r128(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr[p.stderr.rfind("Summary:"):]
    get = lambda key: float(re.search(key + r"\s*(-?[\d.]+|-inf)", txt).group(1)) \
        if re.search(key + r"\s*(-?[\d.]+)", txt) else None
    return {"I_lufs": get(r"I:"), "LRA_lu": get(r"LRA:"), "true_peak_dbtp": get(r"Peak:")}


def short_term(x, win=3.0, hop=0.5):
    y = k_weight(x)
    p = np.sum(y ** 2, axis=0)
    w, h = int(win * SR), int(hop * SR)
    cs = np.concatenate([[0], np.cumsum(p)])
    starts = np.arange(0, max(1, len(p) - w), h)
    ms = (cs[starts + w] - cs[starts]) / w
    return starts / SR + win / 2, -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))


def stft_mag(mono, n_fft=2048, hop=512):
    f, t, Z = signal.stft(mono, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    return f, t, np.abs(Z)


def onset_envelope(mono):
    f, t, M = stft_mag(mono, 1024, 256)
    L = np.log1p(100 * M)
    flux = np.maximum(0, np.diff(L, axis=1)).sum(axis=0)
    flux = flux - signal.medfilt(flux, 31)
    return t[1:], np.maximum(flux, 0)


def tempo_estimate(t, env, lo=60, hi=180):
    fps = 1 / (t[1] - t[0])
    e = env - env.mean()
    ac = signal.fftconvolve(e, e[::-1])[len(e) - 1:]
    lags = np.arange(len(ac)) / fps
    bpm = 60 / np.maximum(lags, 1e-9)
    m = (bpm >= lo) & (bpm <= hi)
    i = np.argmax(ac[m])
    clarity = float(ac[m][i] / (np.abs(ac[m]).mean() + 1e-9))
    return float(bpm[m][i]), clarity


def grid_histogram(t, env, bar_starts, steps=16):
    """Fold onset strength onto step positions within each bar."""
    h = np.zeros(steps)
    bar_starts = np.asarray(bar_starts)
    for ti, e in zip(t, env):
        b = np.searchsorted(bar_starts, ti) - 1
        if b < 0 or b >= len(bar_starts) - 1:
            continue
        L = bar_starts[b + 1] - bar_starts[b]
        pos = (ti - bar_starts[b]) / L * steps
        h[int(round(pos)) % steps] += e
    return h / (h.sum() + 1e-12)


def chroma_and_bands(x, windows):
    """Per-window features: 12-bin chroma (110-4200 Hz, C = 0) and 24 log-spaced band levels.

    Each feature dimension is standardised across windows before cosine similarity, so the
    matrix shows contrast between sections rather than the shared bass-heavy average.
    """
    mono = x.mean(axis=0)
    f, t, M = stft_mag(mono, 4096, 1024)
    pc_bins = (f > 110) & (f < 4200)
    pcs = ((np.round(12 * np.log2(f[pc_bins] / 440.0)) + 9) % 12).astype(int)
    edges = np.geomspace(30, 16000, 25)
    feats = []
    for a, b in windows:
        cols = (t >= a) & (t < b)
        if not cols.any():
            feats.append(np.zeros(12 + 24))
            continue
        S = M[:, cols].mean(axis=1)
        ch = np.bincount(pcs, weights=S[pc_bins], minlength=12)
        ch = ch / (ch.sum() + 1e-9)
        be = np.array([S[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
        be = 10 * np.log10(be + 1e-9)
        feats.append(np.concatenate([ch, be]))
    F = np.array(feats)
    Z = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    Z[:, :12] *= np.sqrt(24 / 12)          # equal say for harmony and timbre
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    return F, Zn @ Zn.T


def analyse(x, wav_path, song, bar_starts=None):
    mono = x.mean(axis=0)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    nz = np.abs(mono) > 1e-5
    first, last = np.argmax(nz), len(nz) - np.argmax(nz[::-1])
    out = {"duration_s": round(x.shape[1] / SR, 2),
           "audible_s": round((last - first) / SR, 2),
           "sample_peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
           "clipped_samples": int(np.sum(np.abs(x) >= 0.9999)),
           "crest_db": round(20 * np.log10(peak / (rms + 1e-12)), 2),
           "lufs_internal": round(lufs(x), 2)}
    out["ffmpeg"] = ffmpeg_r128(wav_path)
    st_t, st_l = short_term(x)
    active = st_l > -60
    out["short_term_lufs_p10_p95"] = [round(float(np.percentile(st_l[active], 10)), 1),
                                      round(float(np.percentile(st_l[active], 95)), 1)]
    # spectral balance (Welch on mid)
    f, psd = signal.welch(mono, SR, nperseg=8192)
    tot = psd[(f >= 20)].sum()
    out["band_share_pct"] = {name: round(100 * psd[(f >= lo) & (f < hi)].sum() / tot, 1)
                             for name, lo, hi in BANDS}
    out["spectral_centroid_hz"] = round(float((f * psd).sum() / psd.sum()), 0)
    # stereo
    L, R = x
    out["stereo_corr"] = round(float(np.corrcoef(L, R)[0, 1]), 3)
    lo = butter(x, "low", 150, 4)
    out["stereo_corr_below_150"] = round(float(np.corrcoef(lo[0], lo[1])[0, 1]), 3)
    side = (L - R) / 2
    out["side_to_mid_db"] = round(10 * np.log10(np.mean(side ** 2) / (np.mean(((L + R) / 2) ** 2)
                                                                        + 1e-12) + 1e-12), 1)
    # rhythm
    ot, env = onset_envelope(mono)
    bpm, clarity = tempo_estimate(ot, env)
    out["tempo_est_bpm"] = round(bpm, 1)
    out["pulse_clarity"] = round(clarity, 2)
    if bar_starts is not None:
        h = grid_histogram(ot, env, bar_starts)
        out["onset_grid"] = [round(float(v), 3) for v in h]
        out["offbeat_onset_share"] = round(float(h[[i for i in range(16) if i % 4]].sum()), 3)
        windows = list(zip(bar_starts[:-1], bar_starts[1:]))
    else:
        edges = np.arange(0, song.length, 4.0)
        windows = list(zip(edges[:-1], edges[1:]))
    F, S = chroma_and_bands(x, windows)
    n = len(S)
    out["selfsim_adjacent_mean"] = round(float(np.mean([S[i, i + 1] for i in range(n - 1)])), 3)
    far = [S[i, j] for i in range(n) for j in range(i + 8, n)]
    # share of window pairs at least 8 windows apart that are near-identical: literal repetition
    out["literal_repeat_share"] = round(float(np.mean(np.array(far) > 0.8)), 3) if far else None
    out["section_contrast"] = round(float(1 - np.mean(far)), 3) if far else None
    # dominant pitch classes
    ch = F[:, :12].mean(axis=0)
    names = "C C# D Eb E F F# G Ab A Bb B".split()
    out["top_pitch_classes"] = [names[i] for i in np.argsort(ch)[::-1][:5]]
    extra = {"st": (st_t, st_l), "S": S, "windows": windows,
             "onset": (ot, env), "grid": out.get("onset_grid")}
    return out, extra


LEVEL_BANDS = {"low_db": (40, 150), "mid_db": (300, 4000)}


def stem_levels(stems, win=0.5):
    """Each track's level against the pre-master mix, measured only where the track plays.

    Returns {track: {active_s, first_s, rel_db, low_db, mid_db}}, loudest first. rel_db is
    broadband; low_db (40-150 Hz) says who owns the low end; mid_db (300 Hz-4 kHz) says whether a
    hook or riff can read over the rest. 0 dB means the track is as loud as everything together.
    """
    if not stems:
        return {}
    mono = {k: np.asarray(x, dtype=np.float64).mean(axis=0) for k, x in stems.items()}
    n = max(len(x) for x in mono.values())
    mono = {k: np.pad(x, (0, n - len(x))) for k, x in mono.items()}
    w = int(win * SR)
    nw = n // w

    def frames(x):
        return np.sqrt(np.mean(x[: nw * w].reshape(nw, w) ** 2, axis=1) + 1e-20)

    mix = sum(mono.values())
    sos = {b: signal.butter(4, f, btype="band", fs=SR, output="sos")
           for b, f in LEVEL_BANDS.items()}
    mix_f = frames(mix)
    mix_b = {b: frames(signal.sosfilt(s, mix)) for b, s in sos.items()}
    rel = lambda a, b, m: round(float(20 * np.log10(np.sqrt(np.mean(a[m] ** 2))
                                                    / np.sqrt(np.mean(b[m] ** 2)))), 1)
    out = {}
    for k, x in mono.items():
        f = frames(x)
        act = f > 10 ** (-60 / 20) * f.max()
        if not act.any():
            continue
        row = {"active_s": float(act.sum() * win), "first_s": float(np.argmax(act) * win),
               "rel_db": rel(f, mix_f, act)}
        for b, s in sos.items():
            row[b] = rel(frames(signal.sosfilt(s, x)), mix_b[b], act)
        out[k] = row
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["rel_db"]))


def activity_matrix(song, res=1.0):
    """Track x time activity (fraction of each window covered by sounding notes)."""
    edges = np.arange(0, song.length + res, res)
    A = np.zeros((len(song.tracks), len(edges) - 1))
    for i, tr in enumerate(song.tracks):
        for nt in tr.notes:
            a, b = nt.t, nt.t + max(nt.dur, 0.08)
            ia, ib = int(a // res), int(min(b, song.length) // res)
            for k in range(ia, min(ib + 1, A.shape[1])):
                lo, hi = max(a, k * res), min(b, (k + 1) * res)
                A[i, k] += max(0, hi - lo) / res
    return np.clip(A, 0, 1), edges


def figure(x, song, report, extra, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 0.8, 1.2, 1.2])
    mono = x.mean(axis=0)
    ax = fig.add_subplot(gs[0, :])
    f, t, M = stft_mag(mono, 4096, 1024)
    keep = (f >= 30) & (f <= 16000)
    ax.pcolormesh(t, f[keep], 20 * np.log10(M[keep] + 1e-7), shading="auto", vmin=-90, vmax=-10,
                  cmap="magma")
    ax.set_yscale("log"); ax.set_ylim(30, 16000); ax.set_ylabel("Hz")
    ax.set_title(f"{song.title} rev {song.rev}: spectrogram")
    for name, a, b in song.sections:
        ax.axvline(a, color="w", lw=0.6, alpha=0.6)
        ax.text(a + 0.3, 12000, name, color="w", fontsize=7, va="top")

    ax = fig.add_subplot(gs[1, :])
    st_t, st_l = extra["st"]
    ax.plot(st_t, st_l, color="k")
    ax.set_ylabel("short-term LUFS"); ax.set_ylim(max(-50, st_l.min() - 2), st_l.max() + 2)
    ax.set_xlim(0, x.shape[1] / SR)
    for name, a, b in song.sections:
        ax.axvspan(a, b, alpha=0.08, color="C%d" % (hash(name) % 8))
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    A, edges = activity_matrix(song)
    ax.imshow(A, aspect="auto", cmap="Greys", extent=[0, edges[-1], len(song.tracks) - 0.5, -0.5],
              interpolation="nearest")
    ax.set_yticks(range(len(song.tracks)))
    ax.set_yticklabels([t.name for t in song.tracks], fontsize=7)
    ax.set_title("arrangement (note coverage per second)")
    ax.set_xlim(0, x.shape[1] / SR)

    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(extra["S"], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title("bar-by-bar self-similarity"); ax.set_xlabel("bar"); ax.set_ylabel("bar")

    ax = fig.add_subplot(gs[3, 1])
    if extra["grid"]:
        g = extra["grid"]
        ax.bar(range(16), g, color=["C3" if i % 4 == 0 else "C0" for i in range(16)])
        ax.set_xticks(range(16)); ax.set_title("onset strength by 16th position")
    else:
        ot, env = extra["onset"]
        ax.plot(ot, env, lw=0.4); ax.set_title("onset envelope")

    ax = fig.add_subplot(gs[3, 2])
    bs = report["band_share_pct"]
    ax.bar(list(bs), list(bs.values()), color="C2")
    ax.set_title("band share %"); ax.tick_params(axis="x", labelsize=7)
    r = report["ffmpeg"]
    fig.text(0.01, 0.005, f"I {r['I_lufs']} LUFS  LRA {r['LRA_lu']} LU  TP {r['true_peak_dbtp']} dBTP"
             f"  crest {report['crest_db']} dB  clipped {report['clipped_samples']}  "
             f"tempo est {report['tempo_est_bpm']}  corr {report['stereo_corr']}  "
             f"adj-sim {report['selfsim_adjacent_mean']}  repeat {report['literal_repeat_share']}"
             f"  contrast {report['section_contrast']}  pcs {' '.join(report['top_pitch_classes'])}",
             fontsize=9)
    fig.tight_layout(rect=[0, 0.015, 1, 1])
    fig.savefig(path, dpi=80)
    plt.close(fig)
