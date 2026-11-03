"""Deterministic tools the graph calls: render, fingerprint, encode, analysis, file editing."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import PROJ, ROOT, PIECES, HARD_GATES, SOFT_GATES, RATINGS


def out_dir(key, rev):
    return PROJ / "out" / PIECES[key]["slug"] / f"v{rev}"


def wav_path(key, rev):
    return out_dir(key, rev) / f"{PIECES[key]['slug']}_v{rev}.wav"


def latest_rev(key):
    base = PROJ / "out" / PIECES[key]["slug"]
    revs = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()
            and (p / f"{PIECES[key]['slug']}_{p.name}.wav").exists()]
    return max(revs) if revs else 0


def read_analysis(key, rev):
    p = out_dir(key, rev) / "analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _render_cmd(key, rev, job, resume, *extra):
    """render.py for one revision; through the shared CPU budget when the workspace has one
    (scripts/cpu_budget.py beside this project), otherwise directly."""
    direct = [sys.executable, str(PROJ / "render.py"), key, str(rev), *extra]
    budget = ROOT / "scripts" / "cpu_budget.py"
    if not budget.exists():
        return direct, PROJ
    return [sys.executable, str(budget), "run", "--session", "music", "--job", job,
            "--cores", "1", "--resume", resume, "--", *direct], ROOT


def render(key, rev):
    """Render through the shared CPU budget. A previous render of the same revision is kept in
    v<rev>/attempts/ first, so no heard or praised render is ever overwritten."""
    d = out_dir(key, rev)
    w = wav_path(key, rev)
    if w.exists():
        att = d / "attempts"
        att.mkdir(parents=True, exist_ok=True)
        k = len(list(att.glob("try*"))) + 1
        keep = att / f"try{k}"
        keep.mkdir()
        for f in d.iterdir():
            if f.is_file():
                shutil.copy2(f, keep / f.name)
    cmd, cwd = _render_cmd(key, rev, f"music agent {key} v{rev}",
                           f"cd instrumentals && python render.py {key} {rev}")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr or p.stdout)[-3000:]}
    return {"ok": True, "analysis": read_analysis(key, rev)}


PREVIEW_KEYS = ("ffmpeg", "short_term_lufs_p10_p95", "band_share_pct", "spectral_centroid_hz",
                "crest_db", "stereo_corr", "tempo_est_bpm", "pulse_clarity",
                "offbeat_onset_share", "literal_repeat_share", "section_contrast")


def render_preview(key, rev):
    """Draft-render revision rev into out/<slug>/preview/ (overwritten each time) and return the
    measurements the composer mixes by, including every track's level against the mix."""
    cmd, cwd = _render_cmd(key, rev, f"music agent {key} v{rev} preview",
                           "draft render; rerun from the composer if needed", "--preview")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return f"Render failed:\n{(p.stderr or p.stdout)[-3000:]}"
    a = json.loads((PROJ / "out" / PIECES[key]["slug"] / "preview" / "analysis.json")
                   .read_text(encoding="utf-8"))
    head = {k: a[k] for k in PREVIEW_KEYS if k in a}
    head["limiter_max_gr_db"] = a["render"].get("limiter_max_gr_db")
    head["master_gain_db"] = a["render"].get("master_gain_db")
    rows = ["track         active_s  first_s  rel_db  low_db  mid_db"]
    for name, r in a.get("stem_levels", {}).items():
        rows.append(f"{name:13s} {r['active_s']:8.1f} {r['first_s']:8.1f} {r['rel_db']:7.1f} "
                    f"{r['low_db']:7.1f} {r['mid_db']:7.1f}")
    return (f"v{rev} draft rendered ({a['audible_s']} s audible).\n{json.dumps(head)}\n\n"
            "Track levels against the whole pre-master mix, measured only while each track "
            "plays (0 dB = as loud as everything together; low = 40-150 Hz, mid = 300 Hz-4 "
            "kHz):\n" + "\n".join(rows))


def gates(key, a):
    """Hard gates decide whether a render counts; soft gates are reported to the critic."""
    hard, soft = [], []
    tp = a["ffmpeg"]["true_peak_dbtp"]
    if tp is None or tp > HARD_GATES["true_peak_dbtp_max"]:
        hard.append(f"true peak {tp} dBTP above {HARD_GATES['true_peak_dbtp_max']}")
    if a["clipped_samples"] > HARD_GATES["clipped_samples_max"]:
        hard.append(f"{a['clipped_samples']} clipped samples")
    if not HARD_GATES["audible_s_min"] <= a["audible_s"] <= HARD_GATES["audible_s_max"]:
        hard.append(f"audible length {a['audible_s']} s outside 90-180 s")
    gr = a["render"].get("limiter_max_gr_db", 0)
    if gr > SOFT_GATES["limiter_gr_db_max"]:
        soft.append(f"limiter takes {gr} dB (soft limit {SOFT_GATES['limiter_gr_db_max']})")
    lu = a["ffmpeg"]["I_lufs"]
    if lu is not None and abs(lu - PIECES[key]["target"]) > SOFT_GATES["lufs_tolerance"]:
        soft.append(f"loudness {lu} LUFS, target {PIECES[key]['target']}")
    return {"hard_pass": not hard, "hard": hard, "soft": soft}


FINGERPRINT_SCRIPT = r'''
import hashlib, importlib, json, sys
from dataclasses import asdict
sys.path.insert(0, sys.argv[1])
from engine.song import song_config
mod = importlib.import_module(sys.argv[2])
out = {}
for rev in range(1, int(sys.argv[3]) + 1):
    s = mod.build(rev)
    cfg = song_config(s)
    cfg.pop("rev", None)
    blob = json.dumps([cfg, {t.name: [asdict(n) for n in t.notes] for t in s.tracks}],
                      sort_keys=True, default=str)
    out[rev] = hashlib.sha256(blob.encode()).hexdigest()
print(json.dumps(out))
'''


def fingerprint(key, upto):
    """Hash of every revision's full score and configuration (excluding its notes text), built
    in a fresh process. Used to prove an edit leaves earlier revisions untouched."""
    mod = PIECES[key]["file"].replace("/", ".").removesuffix(".py")
    script = PROJ / "agent" / "state" / "_fingerprint.py"
    script.write_text(FINGERPRINT_SCRIPT, encoding="utf-8")
    p = subprocess.run([sys.executable, str(script), str(PROJ), mod, str(upto)], cwd=PROJ,
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return {"ok": False, "error": p.stderr[-3000:]}
    return {"ok": True, "hashes": json.loads(p.stdout.strip().splitlines()[-1])}


def to_mp3(wav, mp3, kbps=160):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "2", "-b:a",
                    f"{kbps}k", str(mp3)], check=True)
    return mp3


def ratings(key):
    """Jonathan's own ratings of earlier revisions (the ground truth the agent steers by)."""
    if not RATINGS.exists():
        return {}
    return json.loads(RATINGS.read_text(encoding="utf-8")).get(PIECES[key]["slug"], {})


def add_rating(key, rev, score, note):
    data = json.loads(RATINGS.read_text(encoding="utf-8")) if RATINGS.exists() else {}
    data.setdefault(PIECES[key]["slug"], {})[str(rev)] = {"score": score, "note": note}
    RATINGS.write_text(json.dumps(data, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- confined text editor

class Editor:
    """Client side of Anthropic's text editor tool. Reads anywhere inside instrumentals/
    (except secrets and audio); writes only to the one piece file being revised."""

    def __init__(self, key):
        self.writable = (PROJ / PIECES[key]["file"]).resolve()
        self.backup = self.writable.read_text(encoding="utf-8")

    def _path(self, raw):
        p = Path(raw)
        p = (PROJ / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_relative_to(PROJ) or p.name == ".env" or p.suffix in (".wav", ".mp3"):
            raise PermissionError(f"{raw} is outside what this tool may read")
        return p

    def run(self, inp):
        cmd = inp.get("command")
        p = self._path(inp.get("path", ""))
        if cmd == "view":
            if p.is_dir():
                return "\n".join(sorted(str(q.relative_to(PROJ)) for q in p.iterdir()))
            lines = p.read_text(encoding="utf-8").splitlines()
            lo, hi = 1, len(lines)
            if inp.get("view_range"):
                lo, hi = inp["view_range"]
                hi = len(lines) if hi == -1 else hi
            return "\n".join(f"{i}\t{lines[i - 1]}" for i in range(lo, min(hi, len(lines)) + 1))
        if p != self.writable:
            raise PermissionError(f"only {self.writable.relative_to(PROJ)} may be edited")
        text = p.read_text(encoding="utf-8")
        if cmd == "str_replace":
            old, new = inp["old_str"], inp.get("new_str", "")
            n = text.count(old)
            if n != 1:
                raise ValueError(f"old_str matches {n} times; it must match exactly once")
            p.write_text(text.replace(old, new), encoding="utf-8")
            return "edited"
        if cmd == "insert":
            line = int(inp["insert_line"])
            new = inp.get("insert_text", inp.get("new_str", ""))
            lines = text.splitlines(keepends=True)
            lines.insert(line, new if new.endswith("\n") else new + "\n")
            p.write_text("".join(lines), encoding="utf-8")
            return "inserted"
        if cmd == "create":
            p.write_text(inp["file_text"], encoding="utf-8")
            return "written"
        raise ValueError(f"unsupported command {cmd}")

    def revert(self):
        self.writable.write_text(self.backup, encoding="utf-8")
