"""Render a piece at a revision: WAV, MIDI, score, patches, analysis and diagnostic figure.

    python render.py a 1          # piece A, revision 1
    python render.py all 3        # every piece at revision 3
    python render.py a 2 --stems  # also write per-track stems (24-bit WAV)
    python render.py c 6 --preview  # draft render into out/<slug>/preview/ (overwritten)

Everything is deterministic: same piece, revision and code give the same audio.
"""
import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import song as S          # noqa: E402
from engine import analysis as A      # noqa: E402

PIECES = {"a": "pieces.a_rivet_logic", "b": "pieces.b_lantern_arithmetic",
          "c": "pieces.c_concrete_tide"}


def run(key, rev, stems=False, analyse_only=False, preview=False):
    mod = importlib.import_module(PIECES[key])
    t0 = time.time()
    song = mod.build(rev)
    out = ROOT / "out" / song.slug / ("preview" if preview else f"v{rev}")
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{song.slug}_v{rev}.wav"
    if analyse_only:
        master, st = S.read_wav(wav), {}
        info = json.loads((out / "analysis.json").read_text())["render"]
    else:
        # stems are references to the per-track signals, so per-track levels cost little
        master, info, st = S.render(song, stems=True)
        S.write_wav(wav, master, 24)
    t_render = time.time() - t0
    if not analyse_only:
        # composition files are written only with the audio, so they always describe the WAV
        S.write_midi(out / f"{song.slug}_v{rev}.mid", song)
        S.save_json(out / "config.json", S.song_config(song))
        S.save_json(out / "score.json",
                    {tr.name: [asdict(n) for n in tr.notes] for tr in song.tracks})
    if stems:
        sd = out / "stems"
        sd.mkdir(exist_ok=True)
        for name, x in st.items():
            S.write_wav(sd / f"{name}.wav", x / max(1.0, abs(x).max()), 24)
    bar_starts = song.meta.get("bar_starts")
    report, extra = A.analyse(master, wav, song, bar_starts)
    report["render"] = info if analyse_only else info | {"seconds": round(t_render, 1)}
    if st:
        report["stem_levels"] = A.stem_levels(st)
    elif analyse_only and (out / "analysis.json").exists():
        old = json.loads((out / "analysis.json").read_text())
        if "stem_levels" in old:
            report["stem_levels"] = old["stem_levels"]
    report["revision_notes"] = song.notes_md
    S.save_json(out / "analysis.json", report)
    A.figure(master, song, report, extra, out / "analysis.png")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("onset_grid", "revision_notes", "stem_levels")},
                     indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("piece", choices=list(PIECES) + ["all"])
    ap.add_argument("rev", type=int)
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="draft render into out/<slug>/preview/, leaving numbered versions alone")
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-run analysis on the existing WAV without rendering")
    a = ap.parse_args()
    for k in (PIECES if a.piece == "all" else [a.piece]):
        run(k, a.rev, a.stems, a.analyse_only, a.preview)
