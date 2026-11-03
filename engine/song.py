"""Song model, renderer, mixer, master chain and MIDI writer."""
import json
import struct
from dataclasses import dataclass, field, asdict

import numpy as np

from . import fx
from .dsp import SR, n_of, filt, butter, pan_stereo, place, undb, fade_tail
from .instruments import VOICES


@dataclass
class Note:
    t: float            # start, seconds
    dur: float          # gate, seconds
    pitch: float        # MIDI number (GM drum number on drum tracks)
    vel: float = 0.8
    params: dict = field(default_factory=dict)


@dataclass
class Track:
    name: str
    voice: str
    patch: dict
    notes: list = field(default_factory=list)
    gain_db: float = 0.0
    pan: float = 0.0
    stereo: dict = None          # {"chorus": {...}} or {"haas": ms}; None = point source
    sends: dict = field(default_factory=dict)      # bus -> linear send level
    inserts: list = field(default_factory=list)    # [(name, kwargs)]
    auto: list = None            # gain automation [(t, dB)], linear in dB
    duck: tuple = None           # (key track name, depth dB, release s)
    channel: int = 0
    program: int = 0             # GM program number for the MIDI file


@dataclass
class Song:
    title: str
    slug: str
    bpm: float
    length: float                # seconds of music (tails are added)
    tracks: list
    buses: dict
    master: dict
    sections: list               # [(name, start_s, end_s)]
    seed: int
    rev: int
    notes_md: str = ""           # what this revision changed (for the diary)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- inserts

def _automation(points, n, default=0.0):
    if not points:
        return np.full(n, default)
    ts = np.array([p[0] for p in points]) * SR
    vs = np.array([p[1] for p in points], float)
    return np.interp(np.arange(n), ts, vs)


# Pieces may register their own effects here (name -> fn(x, rng, **kw)) and their own voices in
# instruments.VOICES, so a piece can grow new sounds without changing any other piece.
CUSTOM_INSERTS = {}


def apply_insert(x, name, kw, rng):
    if name in CUSTOM_INSERTS:
        return CUSTOM_INSERTS[name](x, rng, **kw)
    if name == "tape":
        return fx.tape(x, rng=rng, **kw)
    if name == "chorus":
        return fx.chorus(x, rng=rng, **kw)
    if name == "ringmod":
        mix = _automation(kw["mix_points"], x.shape[-1]) if "mix_points" in kw else kw.get("mix", 1.0)
        return fx.ringmod(x, kw["freq"], mix)
    if name == "decimate":
        return fx.decimate_hold(x, kw["hold"], kw.get("bits"))
    if name == "comp":
        return fx.compress(x, **kw)
    if name == "eq":
        y = x
        for band in kw["bands"]:
            y = filt(y, band[0], band[1], band[2] if len(band) > 2 else 0.707,
                     band[3] if len(band) > 3 else 0.0)
        return y
    if name == "hp":
        return butter(x, "high", kw["f"], kw.get("order", 2))
    if name == "lp":
        return butter(x, "low", kw["f"], kw.get("order", 2))
    if name == "drive":
        g = kw.get("gain", 2.0)
        y = np.tanh(g * x + kw.get("bias", 0.0)) - np.tanh(kw.get("bias", 0.0))
        return y / np.tanh(g) * kw.get("out", 1.0)
    if name == "fuzz":
        g = kw.get("gain", 8.0)
        y = np.tanh(g * x)
        y = np.sign(y) * np.abs(y) ** kw.get("shape", 0.7)
        return butter(y, "low", kw.get("tone", 4500), 2) * kw.get("out", 0.5)
    if name == "softclip":
        # shaves the top `shave_db` of transient peaks smoothly, relative to the track's own peak
        pk = np.max(np.abs(x)) + 1e-12
        t = pk * undb(-kw.get("shave_db", 3.0)) * kw.get("knee", 0.7)
        a = np.abs(x)
        over = np.maximum(a - t, 0)
        room = pk * undb(-kw.get("shave_db", 3.0)) - t
        y = np.where(a > t, t + room * np.tanh(over / (room + 1e-12)), a)
        return np.sign(x) * y
    if name == "lp_auto":
        from .dsp import filt_tv
        f = _automation(kw["points"], x.shape[-1], 20000)
        if x.ndim == 1:
            return filt_tv(x, "lp", f, kw.get("q", 0.8), block=128)
        return np.stack([filt_tv(c, "lp", f, kw.get("q", 0.8), block=128) for c in x])
    if name == "varispeed":
        r = _automation(kw["points"], x.shape[-1], 0.0)   # points in cents
        return fx.varispeed(x, 2 ** (r / 1200))
    if name == "gate":
        # rhythmic gate: open on listed (start, end) windows, smoothed
        g = np.zeros(x.shape[-1])
        for a, b in kw["windows"]:
            g[n_of(a):n_of(b)] = 1.0
        g = np.convolve(g, np.ones(n_of(kw.get("smooth", 0.004))) / n_of(kw.get("smooth", 0.004)),
                        mode="same")
        return x * g
    raise ValueError(name)


# ---------------------------------------------------------------- rendering

def render_track(tr, total_n, rng):
    """Render notes, inserts and placement. Returns (stereo dry, mono pre-pan) arrays."""
    voice = VOICES[tr.voice]
    buf = np.zeros(total_n)
    for note in tr.notes:
        sig = voice(note, tr.patch, rng)
        sig = fade_tail(sig, 0.003)
        start = n_of(note.t)
        if note.params.get("reverse"):
            sig = sig[::-1]
            start = n_of(note.t + note.dur) - len(sig)
        place(buf, sig * note.params.get("gain", 1.0), start)
    x = buf
    for name, kw in tr.inserts:
        x = apply_insert(x, name, kw, rng)
    if x.ndim == 1:
        if tr.stereo and "chorus" in tr.stereo:
            st = fx.chorus(x, rng=rng, **tr.stereo["chorus"])
            st = pan_stereo(np.ones(1), tr.pan) * 1.4142 * st
        elif tr.stereo and "haas" in tr.stereo:
            d = n_of(tr.stereo["haas"] / 1000)
            r = np.concatenate([np.zeros(d), x[:-d]])
            st = np.stack([x, r]) * 0.75
        else:
            st = pan_stereo(x, tr.pan)
    else:
        st = x
    g = undb(tr.gain_db)
    if tr.auto:
        g = g * undb(_automation(tr.auto, total_n, 0.0))
    return st * g


def k_weight(x):
    y = filt(x, "highshelf", 1681.97, 0.7072, 3.99984)
    return filt(y, "hp", 38.135, 0.5003)


def lufs(x):
    """Integrated loudness per ITU-R BS.1770-4 (gated). x stereo (2, n)."""
    y = k_weight(x)
    blk, hop = n_of(0.4), n_of(0.1)
    p = np.sum(y ** 2, axis=0)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, len(p) - blk, hop)
    ms = (cs[starts + blk] - cs[starts]) / blk
    l = -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))
    ms = ms[l > -70]
    if not len(ms):
        return -70.0
    rel = -0.691 + 10 * np.log10(ms.mean()) - 10
    l2 = -0.691 + 10 * np.log10(ms)
    return float(-0.691 + 10 * np.log10(ms[l2 > rel].mean()))


def render(song, stems=False):
    """Render a Song to a stereo master. Returns (master, info, stem dict)."""
    # noise draws are tied to revision 1 so revisions differ only by their stated changes
    rng_root = np.random.SeedSequence([song.seed, 1])
    tail = song.master.get("tail", 4.0)
    total_n = n_of(song.length + tail)
    dry = {}
    for i, tr in enumerate(song.tracks):
        rng = np.random.default_rng(rng_root.spawn(1)[0])
        dry[tr.name] = render_track(tr, total_n, rng)
    # side-chain ducking (M1): key on the key track's pre-fader energy
    for tr in song.tracks:
        if tr.duck:
            key, depth, rel = tr.duck
            env = fx.envelope(dry[key], 0.002, rel)
            env = env / (env.max() + 1e-9)
            dry[tr.name] = dry[tr.name] * undb(-depth * np.clip(env * 3, 0, 1))
    mix = np.zeros((2, total_n))
    for tr in song.tracks:
        mix += dry[tr.name]
    returns = {}
    for bname, b in song.buses.items():
        send = np.zeros((2, total_n))
        for tr in song.tracks:
            lvl = tr.sends.get(bname, 0.0)
            if lvl:
                send += dry[tr.name] * lvl
        rng = np.random.default_rng(rng_root.spawn(1)[0])
        if b["type"] == "reverb":
            ir = fx.make_ir(rng=rng, **b.get("ir", {}))
            wet = fx.convolve_ir(send, ir)
        elif b["type"] == "echo":
            wet = fx.echo(send, rng=rng, **b.get("echo", {}))
        else:
            raise ValueError(b["type"])
        for name, kw in b.get("inserts", []):
            wet = apply_insert(wet, name, kw, rng)
        wet = wet * undb(b.get("return_db", 0.0))
        returns[bname] = wet
        mix += wet
    master, info = master_chain(mix, song.master, np.random.default_rng(rng_root.spawn(1)[0]))
    out_stems = {}
    if stems:
        out_stems = {**dry, **{"bus_" + k: v for k, v in returns.items()}}
    return master, info, out_stems


def master_chain(x, m, rng):
    y = fx.mono_below(x, m.get("mono_below", 120.0), m.get("linear_phase_mono", False))
    for band in m.get("eq", []):
        y = filt(y, band[0], band[1], band[2] if len(band) > 2 else 0.707,
                 band[3] if len(band) > 3 else 0.0)
    if m.get("comp"):
        y = fx.compress(y, **m["comp"])
    if m.get("tape"):
        y = fx.tape(y, rng=rng, **m["tape"])
    target = m.get("target_lufs", -14.0)
    ceiling = m.get("ceiling", -1.5)
    gain_db = 0.0
    gr_max = 0.0
    for _ in range(3):
        cur = lufs(y * undb(gain_db))
        gain_db += target - cur
        lim, g = fx.limiter(y * undb(gain_db), ceiling, true_peak=m.get("true_peak", False))
        gr_max = float(-20 * np.log10(max(g.min(), 1e-9)))
        out = lim
        if abs(lufs(out) - target) < 0.15:
            break
    # gentle fade over the tail
    fade = m.get("fade_out")
    if fade:
        a, b = n_of(fade[0]), n_of(fade[1])
        env = np.ones(out.shape[-1])
        env[a:b] = np.linspace(1, 0, b - a) ** 2
        env[b:] = 0
        out = out * env
    info = {"master_gain_db": round(gain_db, 2), "limiter_max_gr_db": round(gr_max, 2),
            "lufs_internal": round(lufs(out), 2)}
    return out, info


# ---------------------------------------------------------------- output

def write_wav(path, x, bits=24):
    """Write stereo float (2, n) as PCM WAV (16 or 24 bit) with TPDF dither."""
    x = np.clip(x, -1.0, 1.0)
    q = 2 ** (bits - 1) - 1
    rng = np.random.default_rng(7)
    d = (rng.uniform(-0.5, 0.5, x.shape) + rng.uniform(-0.5, 0.5, x.shape))
    ints = np.clip(np.round(x * q + d), -q - 1, q).astype(np.int32).T.reshape(-1)
    if bits == 16:
        data = ints.astype("<i2").tobytes()
    else:
        b = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        data = b.tobytes()
    nch, bps = 2, bits // 8
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, nch, SR, SR * nch * bps, nch * bps, bits))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def _vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def write_midi(path, song, ppq=480):
    """Standard MIDI file, type 1: a tempo track and one track per Song track."""
    spt = 60.0 / song.bpm / ppq       # seconds per tick

    def track_chunk(events):
        body = b""
        last = 0
        for tick, data in sorted(events, key=lambda e: (e[0], e[1][0] & 0xF0 == 0x90)):
            body += _vlq(tick - last) + data
            last = tick
        body += _vlq(0) + b"\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(body)) + body

    tempo = int(round(60_000_000 / song.bpm))
    t0 = [(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")),
          (0, b"\xff\x03" + _vlq(len(song.title)) + song.title.encode("ascii", "replace")),
          (0, b"\xff\x58\x04\x04\x02\x18\x08")]
    chunks = [track_chunk(t0)]
    for tr in song.tracks:
        ch = 9 if tr.voice == "drums" else tr.channel % 16
        name = tr.name.encode("ascii", "replace")
        ev = [(0, b"\xff\x03" + _vlq(len(name)) + name)]
        if ch != 9:
            ev.append((0, bytes([0xC0 | ch, tr.program & 0x7F])))
        for nt in tr.notes:
            p = int(round(nt.pitch))
            if not 0 <= p <= 127:
                continue
            on = int(round(nt.t / spt))
            off = max(on + 1, int(round((nt.t + nt.dur) / spt)))
            v = int(np.clip(round(nt.vel * 127), 1, 127))
            ev.append((on, bytes([0x90 | ch, p, v])))
            ev.append((off, bytes([0x80 | ch, p, 0])))
        chunks.append(track_chunk(ev))
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ppq))
        for c in chunks:
            f.write(c)


def song_config(song):
    """Everything except the note lists: patches, mix, buses, master, sections."""
    return {
        "title": song.title, "slug": song.slug, "bpm": song.bpm, "rev": song.rev,
        "seed": song.seed, "length_s": song.length, "sections": song.sections,
        "buses": song.buses, "master": song.master, "meta": song.meta,
        "tracks": [{k: v for k, v in asdict(t).items() if k != "notes"} | {"n_notes": len(t.notes)}
                   for t in song.tracks],
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def read_wav(path):
    """Read a 16- or 24-bit PCM stereo WAV written by write_wav into float (2, n)."""
    with open(path, "rb") as f:
        raw = f.read()
    bits = struct.unpack("<H", raw[34:36])[0]
    i = raw.index(b"data") + 8
    data = np.frombuffer(raw[i:], dtype=np.uint8)
    if bits == 16:
        ints = data.view("<i2").astype(np.int32)
    else:
        b = data.reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
    return ints.reshape(-1, 2).T / float(2 ** (bits - 1) - 1)
