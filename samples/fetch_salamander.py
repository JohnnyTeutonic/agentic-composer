"""Fetch the piano samples the engine uses (not kept in Git).

    python samples/fetch_salamander.py

Downloads the Salamander Grand Piano V3 (Alexander Holm, CC-BY 3.0) 44.1 kHz 16-bit archive from
FreePats into a temporary folder, keeps velocity layers 4, 8, 12 and 16 in samples/salamander/,
and deletes the rest.
"""
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = ("https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
       "SalamanderGrandPianoV3+20161209_44khz16bit.tar.xz")
KEEP = re.compile(r"[A-G]#?\dv(4|8|12|16)\.wav")
DEST = Path(__file__).resolve().parent / "salamander"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / "salamander.tar.xz"
        print("downloading", URL)
        urllib.request.urlretrieve(URL, arc)
        with tarfile.open(arc, "r:xz") as t:
            for m in t.getmembers():
                name = Path(m.name).name
                if m.isfile() and (KEEP.fullmatch(name) or name == "README"):
                    src = t.extractfile(m)
                    out = DEST / ("AUTHOR_README.txt" if name == "README" else name)
                    with out.open("wb") as f:
                        shutil.copyfileobj(src, f)
    print(len(list(DEST.glob("*.wav"))), "samples in", DEST)


if __name__ == "__main__":
    main()
