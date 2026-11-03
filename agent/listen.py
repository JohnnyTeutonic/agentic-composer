"""The listening critic: Gemini hears the render (and the previous revision, for an A/B)."""
import json
import os

from . import obs
from .config import GEMINI_MODEL, PIECES, PROJ
from .tools import to_mp3, wav_path

LISTEN_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 1, "maximum": 10} for k in
                           ("groove", "development", "sound_design", "mix", "originality",
                            "overall")},
            "required": ["groove", "development", "sound_design", "mix", "originality",
                         "overall"],
        },
        "vs_previous": {"type": "string", "enum": ["better", "worse", "same", "not_compared"]},
        "vs_previous_reason": {"type": "string"},
        "moments": {"type": "array", "items": {
            "type": "object",
            "properties": {"time_s": {"type": "number"}, "observation": {"type": "string"}},
            "required": ["time_s", "observation"]}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scores", "vs_previous", "vs_previous_reason", "moments", "strengths",
                 "weaknesses"],
}


def _mp3(key, rev):
    d = PROJ / "agent" / "state" / "mp3"
    d.mkdir(parents=True, exist_ok=True)
    mp3 = d / f"{PIECES[key]['slug']}_v{rev}.mp3"
    w = wav_path(key, rev)
    if not mp3.exists() or mp3.stat().st_mtime < w.stat().st_mtime:
        to_mp3(w, mp3)
    return mp3


def listen(key, rev, prev_rev=None):
    from google import genai
    from google.genai import types

    p = PIECES[key]
    parts = []
    if prev_rev:
        parts.append(f"Track 1 is the PREVIOUS version (v{prev_rev}).")
        parts.append(types.Part.from_bytes(data=_mp3(key, prev_rev).read_bytes(),
                                           mime_type="audio/mp3"))
        parts.append(f"Track 2 is the CURRENT version (v{rev}); judge this one.")
    parts.append(types.Part.from_bytes(data=_mp3(key, rev).read_bytes(), mime_type="audio/mp3"))
    parts.append(f"""You are a demanding producer and critic listening to an original
instrumental, "{p['title']}". Brief: {p['brief']}
It is synthesised entirely in code; there are no samples. Its aesthetic references are
production principles of late-1970s and early-1980s art-rock, electronic funk and ambient
records, without copying any of them.
Listen to the whole piece. Score each dimension from 1 to 10 against serious released music in
these genres, not against other code-generated music; a 7 should be rare and earned.
Give timestamped moments (seconds) where something works or fails: grooves that lock or
sag, transitions, sounds that are thin, harsh or muddy, passages that overstay. List concrete
strengths and weaknesses. {"Compare it with the previous version." if prev_rev else
"Set vs_previous to not_compared."} Be specific; do not flatter.""")
    gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    with obs.span("listener", as_type="generation", model=GEMINI_MODEL,
                  input={"rev": rev, "prev_rev": prev_rev}) as g:
        resp = gclient.models.generate_content(
            model=GEMINI_MODEL, contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_json_schema=LISTEN_SCHEMA))
        out = json.loads(resp.text)
        if g:
            um = resp.usage_metadata
            g.update(output=out, usage_details={
                "input_tokens": um.prompt_token_count or 0,
                "output_tokens": um.candidates_token_count or 0})
    for k, v in out["scores"].items():
        obs.score(f"listener.{k}", float(v), comment=f"v{rev}")
    return out
