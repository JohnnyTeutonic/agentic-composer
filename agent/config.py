"""Paths, models, pieces and secrets for the music agent.

Secrets come only from instrumentals/.env (git-ignored) or the process environment. Values are
never printed or logged; `python -m agent.check` reports only whether each service authenticates.
"""
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]          # instrumentals/
ROOT = PROJ.parent                                   # workspace root (cpu_budget.py lives here)
ENV_FILE = PROJ / ".env"
STATE_DB = PROJ / "agent" / "state" / "graph.sqlite"
RATINGS = PROJ / "agent" / "ratings.json"

# Reasoning model for critique and composition (Anthropic API).
CLAUDE_MODEL = os.environ.get("MUSIC_CLAUDE_MODEL", "claude-opus-5-5")
COMPOSE_EFFORT = os.environ.get("MUSIC_COMPOSE_EFFORT", "high")
CRITIQUE_EFFORT = os.environ.get("MUSIC_CRITIQUE_EFFORT", "high")
# Listening model (Gemini, audio input). Confirm with `python -m agent.check`, which lists the
# audio-capable models the key can see; override with MUSIC_GEMINI_MODEL.
GEMINI_MODEL = os.environ.get("MUSIC_GEMINI_MODEL", "gemini-3.1-pro-preview")

# Claude Opus 5.5 list prices (USD per million tokens) for the run's cost guard.
PRICE_IN, PRICE_OUT = 4.00, 20.00

PIECES = {
    "a": {"slug": "a_rivet_logic", "file": "pieces/a_rivet_logic.py", "title": "Rivet Logic",
          "target": -14.0,
          "brief": "Angular electronic funk, 100-120 BPM (here 112): syncopated bass, clipped "
                   "motifs, dry drums, strong rhythmic hooks. 90-180 s."},
    "b": {"slug": "b_lantern_arithmetic", "file": "pieces/b_lantern_arithmetic.py",
          "title": "Lantern Arithmetic", "target": -17.0,
          "brief": "Generative miniature, 65-95 BPM (nominal 76): slowly changing harmonic "
                   "textures, sparse melodic events, irregular timbral transformations. 90-180 s."},
    "c": {"slug": "c_concrete_tide", "file": "pieces/c_concrete_tide.py", "title": "Concrete Tide",
          "target": -14.0,
          "brief": "Hybrid art-rock instrumental (88 BPM): bass-forward, unconventional "
                   "percussion, atmospheric synthesisers, alternating density and space. 90-180 s."},
}

# Hard gates block a render from counting as a revision; soft gates are reported only.
# Jonathan rated a render with 6.5 dB of limiting highly (A v3), so limiting is a soft gate.
HARD_GATES = {"true_peak_dbtp_max": -1.0, "clipped_samples_max": 0,
              "audible_s_min": 90.0, "audible_s_max": 180.0}
SOFT_GATES = {"limiter_gr_db_max": 3.0, "lufs_tolerance": 0.5}


def _load(f):
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v


def load_env():
    """Load KEY=VALUE lines into os.environ from .env (existing values win; empty values are
    skipped). If MUSIC_ENV_FALLBACK names another file (relative to this folder), its keys fill
    in whatever is still missing, so keys kept elsewhere need not be copied."""
    if ENV_FILE.exists():
        _load(ENV_FILE)
    fb = os.environ.get("MUSIC_ENV_FALLBACK")
    if fb:
        p = Path(fb) if Path(fb).is_absolute() else PROJ / fb
        if p.exists():
            _load(p)


def have(*names):
    return all(os.environ.get(n) for n in names)
