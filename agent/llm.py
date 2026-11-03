"""Claude calls: the critic (structured output) and the composer (tool loop editing one file)."""
import json

import anthropic

from . import obs
from .config import CLAUDE_MODEL, COMPOSE_EFFORT, CRITIQUE_EFFORT, PRICE_IN, PRICE_OUT, PIECES, PROJ
from .tools import Editor, fingerprint, render_preview

FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

CONTEXT = """You work on an experimental instrumental music project. Three original instrumentals
are composed as Python code: a numpy synthesis engine (engine/), one file per piece (pieces/),
rendered by render.py into out/<slug>/v<rev>/ with a WAV, MIDI, score.json, config.json,
analysis.json and analysis.png. The aesthetic references are production principles associated
with 1970s and early-1980s art-rock, electronic funk and ambient records: syncopated bass cells,
clipped motifs, dry machine-precise drums, treated synthesisers, tape textures, generative
repetition, arrangement by subtraction, harmonic ambiguity that stays accessible. Never copy any
existing melody, riff, recording or arrangement.
research/production_notes.md defines the rules (R-numbers) and mix rules (M1-M3). engine/instruments.py lists the
voices (synth_bass, blip, fm, pluck, pad, lead, organ, metal, breath, drums) and their patch keys;
engine/sampler.py adds "piano", a sampled grand piano (Salamander, CC-BY 3.0; patch keys
layer_bias, release, max_len, bright, body, tune; velocity picks one of four recorded layers);
engine/song.py lists the inserts (tape, chorus, ringmod, decimate, comp, eq, hp, lp, drive, fuzz,
softclip, lp_auto, varispeed, gate).
Musical interest is the objective. Measurements are guard rails. The only real test is human
listening: the ratings and notes from Jonathan (and any guest rater he brings in; the note says
who) outrank every other input, including the listening model's."""


def _client():
    return anthropic.Anthropic()


def _cost(usage):
    return (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1e6


def _call(**kw):
    resp = _client().beta.messages.create(model=CLAUDE_MODEL, cache_control={"type": "ephemeral"},
                                          **FALLBACK, **kw)
    if resp.stop_reason == "refusal":
        detail = resp.stop_details.explanation if resp.stop_details else ""
        raise RuntimeError(f"model declined the request: {detail}")
    return resp


def _stream(**kw):
    with _client().beta.messages.stream(model=CLAUDE_MODEL, cache_control={"type": "ephemeral"},
                                        **FALLBACK, **kw) as st:
        resp = st.get_final_message()
    if resp.stop_reason == "refusal":
        detail = resp.stop_details.explanation if resp.stop_details else ""
        raise RuntimeError(f"model declined the request: {detail}")
    return resp


def _log(path, turn, resp):
    """One line per composer turn: stop reason, usage, and what the model did."""
    blocks = []
    for b in resp.content:
        if b.type == "text":
            blocks.append({"text": b.text[:600]})
        elif b.type == "tool_use":
            inp = b.input if isinstance(b.input, dict) else {}
            blocks.append({"tool": b.name, "command": inp.get("command"), "path": inp.get("path"),
                           "chars": sum(len(str(v)) for v in inp.values())})
        else:
            blocks.append({"type": b.type})
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"turn": turn, "stop": resp.stop_reason,
                            "usage": _usage_dict(resp.usage), "blocks": blocks}) + "\n")


def _usage_dict(u):
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0}


CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict": {"type": "string", "enum": ["revise", "accept"]},
        "keep": {"type": "array", "items": {"type": "string"}},
        "changes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "area": {"type": "string", "enum": ["groove", "development", "repetition", "form",
                                                    "harmony", "sound_design", "mix"]},
                "change": {"type": "string"},
                "why": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["area", "change", "why", "evidence"],
            "additionalProperties": False}},
    },
    "required": ["summary", "verdict", "keep", "changes"],
    "additionalProperties": False,
}


REVISE_ASK = """Decide whether v{rev} should be revised or accepted. If revising, give 2-5 concrete
changes the composer can implement in the piece file with the existing engine, most important
first, each tied to evidence (listener timestamps, bars, metric values, code). At least two must
be musical (composition, arrangement or sound design), not mix. Name what must be kept: anything
Jonathan praised is untouchable, and changes must not dilute it. Accept when further change is
more likely to harm than help."""

FRESH_ASK = """The human listeners have rejected this piece and asked for it to be abandoned and
started from scratch. Do not revise v{rev}. Write the design for a new composition, v{nxt}, that
answers every point in their notes. Verdict: revise. Put the design in "changes", one item per
component, most important first: (1) the hook (instrument, rhythm, contour and where it is
stated), (2) the bass part and its relation to the kick, (3) the drum beat, bar by bar where it
matters, (4) form and the loud/quiet arc with bar numbers and the tempo, (5) textures and sound
design, (6) the mix: which part leads, and target levels. Use "evidence" for the note or
measurement each item answers. In "keep", list only engine techniques or measurements worth
carrying over; no motif, riff, pattern or section of the rejected piece may be reused. Be
concrete enough that a composer could build it without guessing, and make it distinctive: the
notes call the rejected piece generic."""

FRESH_RULE = """- This is a NEW composition, not a revision: write it in a new function (for example
  build_v{nxt}) that build(rev) hands over to when rev >= {nxt}. Reuse no motif, riff, drum
  pattern, chord progression or section of v1-v{rev}. You may reuse engine voices, effects, and
  sound-building code (custom voices, inserts) from the file where they serve the design.
"""


def critique(key, rev, analysis, listening, history, ratings, human_notes, fresh=False):
    p = PIECES[key]
    code = (PROJ / p["file"]).read_text(encoding="utf-8")
    rules = (PROJ / "research" / "production_notes.md").read_text(encoding="utf-8")
    prompt = f"""Piece {key.upper()}, "{p['title']}". Brief: {p['brief']}
Current revision: v{rev}.

JONATHAN'S NOTES ON THIS REVISION (outrank everything): {human_notes or "none yet"}
JONATHAN'S RATINGS OF EARLIER REVISIONS: {json.dumps(ratings) or "none"}

LISTENING MODEL'S REPORT ON v{rev} (it heard the audio; you cannot):
{json.dumps(listening, indent=1)}

MEASUREMENTS OF v{rev}: {json.dumps({k: v for k, v in (analysis or {}).items() if k != 'onset_grid'})}

EARLIER ROUNDS: {json.dumps(history)}

RULES (research/production_notes.md):
{rules}

PIECE SOURCE ({p['file']}):
{code}

{FRESH_ASK.format(rev=rev, nxt=rev + 1) if fresh else REVISE_ASK.format(rev=rev)}"""
    with obs.span("critic", as_type="generation", model=CLAUDE_MODEL,
                  input={"rev": rev, "human_notes": human_notes}) as g:
        resp = _stream(max_tokens=32000, system=CONTEXT,
                     output_config={"effort": CRITIQUE_EFFORT,
                                    "format": {"type": "json_schema", "schema": CRITIQUE_SCHEMA}},
                     messages=[{"role": "user", "content": prompt}])
        text = next(b.text for b in resp.content if b.type == "text")
        out = json.loads(text)
        if g:
            g.update(output=out, usage_details=_usage_dict(resp.usage))
    return out, _cost(resp.usage)


RENDER_TOOL = {
    "name": "render_and_measure",
    "description": "Render the new revision as it now stands into a draft folder (about two "
                   "minutes) and return its loudness, limiting, frequency balance and every "
                   "track's level against the mix while that track plays. You cannot hear the "
                   "audio; this is how you balance parts. Use it after the arrangement is in "
                   "place and again after rebalancing. At most {n} uses per revision.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}
MAX_RENDERS = 4

CHECK_TOOL = {
    "name": "check_build",
    "description": "Build every revision of the piece in a fresh process and report: whether the "
                   "new revision builds, and whether every earlier revision is still identical "
                   "(score and configuration hashes). Call it after editing and before finishing.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}
EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


def compose(key, rev, next_rev, crit, human_notes, fix_request, ratings, baseline_hashes,
            max_turns=50, fresh=False):
    """Edit the piece file so that build(next_rev) implements the critique. Returns a report."""
    p = PIECES[key]
    editor = Editor(key)
    task = f"""Revise piece {key.upper()}, "{p['title']}" (file {p['file']}) from v{rev} to
v{next_rev}.

JONATHAN'S NOTES (outrank everything): {human_notes or "none"}
JONATHAN'S RATINGS: {json.dumps(ratings) or "none"}
CRITIQUE TO IMPLEMENT: {json.dumps(crit, indent=1)}
{"FIX REQUIRED FIRST: " + fix_request if fix_request else ""}

Rules:
- Edit only {p['file']}, with the str_replace_based_edit_tool. You may view any project file
  (engine/, research/, pieces/, render.py) to learn the API. Do not try to edit anything else.
{FRESH_RULE.format(rev=rev, nxt=next_rev) if fresh else ""}- Gate every change on rev >= {next_rev}, so that revisions 1..{rev} build exactly as before;
  existing branches for earlier revisions stay active. check_build verifies this by hashing
  every earlier revision's score and configuration, and any change to them is rejected.
- You may build new sounds inside the piece file: define voice functions
  fn(note, patch, rng) -> mono numpy array and register them with
  `instruments.VOICES["name"] = fn`, and define effects fn(x, rng, **kw) (x mono or stereo
  (2, n)) and register them with `song.CUSTOM_INSERTS["name"] = fn`. Prefix the names with the
  piece's letter, use them only in rev >= {next_rev} tracks, and draw randomness only from the
  rng argument, so renders stay deterministic. Use this when the existing voices cannot give
  what the notes ask for (lushness, texture, a real guitar tone).
- Add a revision note for {next_rev} to the file's existing notes mechanism, in plain
  Commonwealth-English sentences, saying what changed and why.
- For rev >= {next_rev}, master "true_peak": True and "linear_phase_mono": True must hold.
- Mix by measurement, not estimate. Once the parts are in place, call render_and_measure
  and read the track levels: a hook or riff that must lead should sit within about 6 dB of the
  mix in the mid band while it plays; a part that is meant to be felt can sit lower, but a part
  below about -25 dB is effectively inaudible and should be raised or cut. Keep the limiter's
  gain reduction at 3 dB or less by balancing parts, not by master gain. Rebalance and measure
  again.
- Call check_build before finishing, and fix whatever it reports. Finish with a short plain
  summary of what you changed, quoting the final measured levels of the lead parts."""
    messages = [{"role": "user", "content": task}]
    tools = [EDITOR_TOOL, CHECK_TOOL,
             {**RENDER_TOOL, "description": RENDER_TOOL["description"].format(n=MAX_RENDERS)}]
    cost, turns, renders = 0.0, 0, 0
    with obs.span("composer", as_type="agent", input={"from": rev, "to": next_rev}) as ag:
        log = PROJ / "agent" / "state" / "logs" / f"{key}-v{next_rev}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        nudges = 0
        while turns < max_turns:
            turns += 1
            with obs.span(f"composer-turn-{turns}", as_type="generation",
                          model=CLAUDE_MODEL) as g:
                # streamed with a large ceiling: thinking counts against max_tokens, and a
                # rewrite can need long edits
                resp = _stream(max_tokens=64000, system=CONTEXT, tools=tools,
                               output_config={"effort": COMPOSE_EFFORT}, messages=messages)
                if g:
                    g.update(usage_details=_usage_dict(resp.usage),
                             output=[b.type for b in resp.content])
            cost += _cost(resp.usage)
            _log(log, turns, resp)
            if resp.stop_reason == "max_tokens":
                # the cut-off turn is not kept (its tool call may be incomplete); history stays
                # append-only
                messages.append({"role": "user", "content": "Your last reply reached the output "
                                 "limit before its edit was complete, so it was discarded. Make "
                                 "the change as several smaller str_replace edits."})
                continue
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                status = _check(key, rev, next_rev, baseline_hashes)
                if status.startswith("OK") or nudges >= 2:
                    break
                nudges += 1
                messages.append({"role": "user", "content": f"check_build reports: {status}\n"
                                 "The revision is not finished. Continue editing the piece file."})
                continue
            results = []
            for u in uses:
                try:
                    if u.name == "check_build":
                        out = _check(key, rev, next_rev, baseline_hashes)
                    elif u.name == "render_and_measure":
                        status = _check(key, rev, next_rev, baseline_hashes)
                        if not status.startswith("OK"):
                            out = f"Not rendered; fix the build first. {status}"
                        elif renders >= MAX_RENDERS:
                            out = (f"No renders left ({MAX_RENDERS} used). Finish from the "
                                   "last measurements.")
                        else:
                            renders += 1
                            with obs.span("composer-render", as_type="tool",
                                          input={"rev": next_rev, "n": renders}) as sp:
                                out = render_preview(key, next_rev)
                                if sp:
                                    sp.update(output=out[:2000])
                    else:
                        out = editor.run(u.input)
                    results.append({"type": "tool_result", "tool_use_id": u.id,
                                    "content": out})
                except Exception as e:  # report every tool failure back to the model
                    results.append({"type": "tool_result", "tool_use_id": u.id,
                                    "content": f"Error: {e}", "is_error": True})
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"turn": turns, "tool_results": [
                    str(r["content"])[:300] for r in results]}) + "\n")
            messages.append({"role": "user", "content": results})
        summary = next((b.text for b in reversed(messages[-1]["content"])
                        if getattr(b, "type", None) == "text"), "") \
            if messages[-1]["role"] == "assistant" else ""
        final = _check(key, rev, next_rev, baseline_hashes)
        ok = final.startswith("OK")
        if not ok:
            editor.revert()
        if ag:
            ag.update(output={"ok": ok, "summary": summary, "check": final, "turns": turns})
    return {"ok": ok, "summary": summary, "check": final, "turns": turns,
            "renders": renders}, cost


def _check(key, rev, next_rev, baseline):
    fp = fingerprint(key, next_rev)
    if not fp["ok"]:
        return f"FAILED: the piece does not build:\n{fp['error']}"
    changed = [r for r in range(1, rev + 1) if fp["hashes"][str(r)] != baseline[str(r)]]
    if changed:
        return (f"FAILED: earlier revisions changed: {changed}. Gate the new code on "
                f"rev >= {next_rev}.")
    if fp["hashes"][str(next_rev)] == baseline[str(rev)]:
        return f"FAILED: v{next_rev} is identical to v{rev}; nothing was changed for it."
    return f"OK: v{next_rev} builds and v1-v{rev} are unchanged."
