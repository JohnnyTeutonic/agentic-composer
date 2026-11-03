"""The LangGraph music agent.

    listen -> human -> critique -> compose -> render --> listen -> ... -> END
                 |          |                 |
               (stop)    (accept)      (hard gate failed: back to compose, at most twice)

listen:   Gemini hears the current revision (A/B against the previous one).
human:    in "human" mode the graph pauses (interrupt) for Jonathan's rating and notes; in
          "auto" mode it passes straight through.
critique: Claude reads the listening report, measurements, code, rules and Jonathan's ratings,
          and decides revise or accept.
compose:  Claude edits the piece file through a confined editor; earlier revisions are
          hash-checked so they can never change.
render:   the new revision is rendered through the CPU budget and scored; hard gates are true
          peak, clipping and length. Limiting and loudness are advisory.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from . import obs
from .llm import critique, compose
from .listen import listen
from .tools import gates, read_analysis, render, fingerprint, ratings, add_rating, wav_path


class S(TypedDict, total=False):
    key: str
    mode: str                 # "human" or "auto"
    rev: int                  # current revision (rendered)
    prev_rev: int
    next_rev: int
    max_iters: int
    iteration: int
    budget_usd: float
    cost_usd: float
    listening: dict           # rev -> listener report
    critique: dict
    compose: dict
    human_notes: str
    fix_request: str
    fix_attempts: int
    history: list
    stop_reason: str
    jonathan_stopped: bool
    fresh: bool               # first round writes a new composition instead of revising


def n_listen(s: S):
    key, rev = s["key"], s["rev"]
    with obs.span("listen", input={"rev": rev}):
        if not wav_path(key, rev).exists():
            r = render(key, rev)
            if not r["ok"]:
                raise RuntimeError(r["error"])
        try:
            rep = listen(key, rev, s.get("prev_rev") or None)
        except Exception as e:   # billing, quota or outage: carry on without a listener
            rep = {"unavailable": f"{type(e).__name__}: {str(e)[:300]}",
                   "scores": {}, "vs_previous": "not_compared", "weaknesses": [],
                   "note": "No listening model heard this revision. Rely on Jonathan's notes "
                           "and the measurements, and say so."}
    lis = dict(s.get("listening", {}))
    lis[str(rev)] = rep
    return {"listening": lis}


def n_human(s: S):
    if s["mode"] != "human" or s.get("jonathan_stopped"):
        return {}
    rev = s["rev"]
    rep = s["listening"][str(rev)]
    answer = interrupt({
        "rev": rev, "wav": str(wav_path(s["key"], rev)),
        "listener_scores": rep["scores"], "listener_vs_previous": rep["vs_previous"],
        "listener_weaknesses": rep["weaknesses"][:5],
        "ask": "Rate this revision (any scale or word) and give notes; or stop."})
    notes = answer.get("notes", "")
    if answer.get("rating") is not None:
        add_rating(s["key"], rev, answer["rating"], notes)
        try:
            obs.score("jonathan.rating", float(answer["rating"]), comment=notes)
        except (TypeError, ValueError):
            obs.score("jonathan.rating", str(answer["rating"]), comment=notes,
                      data_type="CATEGORICAL")
    out = {"human_notes": notes}
    if answer.get("stop"):
        out["stop_reason"] = "Jonathan stopped the run"
        out["jonathan_stopped"] = True
    return out


def r_after_human(s: S):
    return END if s.get("stop_reason") else "critique"


def n_critique(s: S):
    key, rev = s["key"], s["rev"]
    with obs.span("critique", input={"rev": rev}):
        crit, cost = critique(key, rev, read_analysis(key, rev), s["listening"][str(rev)],
                              s.get("history", []), ratings(key), s.get("human_notes", ""),
                              fresh=s.get("fresh", False))
    out = {"critique": crit, "cost_usd": s.get("cost_usd", 0.0) + cost}
    if crit["verdict"] == "accept" and not s.get("human_notes"):
        out["stop_reason"] = f"critic accepted v{rev}: {crit['summary']}"
    return out


def r_after_critique(s: S):
    return END if s.get("stop_reason") else "compose"


def n_compose(s: S):
    key, rev = s["key"], s["rev"]
    nxt = rev + 1
    base = fingerprint(key, rev)
    if not base["ok"]:
        raise RuntimeError(f"v{rev} does not build before editing:\n{base['error']}")
    with obs.span("compose", input={"from": rev, "to": nxt}):
        rep, cost = compose(key, rev, nxt, s["critique"], s.get("human_notes", ""),
                            s.get("fix_request", ""), ratings(key), base["hashes"],
                            fresh=s.get("fresh", False))
    out = {"compose": rep, "next_rev": nxt, "cost_usd": s.get("cost_usd", 0.0) + cost,
           "fix_request": ""}
    if not rep["ok"]:
        out["stop_reason"] = f"composer could not produce a valid v{nxt}: {rep['check']}"
    return out


def r_after_compose(s: S):
    return END if s.get("stop_reason") else "render"


def n_render(s: S):
    key, nxt = s["key"], s["next_rev"]
    with obs.span("render", as_type="tool", input={"rev": nxt}) as sp:
        r = render(key, nxt)
        if not r["ok"]:
            return {"stop_reason": f"render of v{nxt} failed: {r['error'][-500:]}"}
        a = r["analysis"]
        g = gates(key, a)
        if sp:
            sp.update(output={"gates": g, "lufs": a["ffmpeg"], "limiter": a["render"]})
    obs.score("gates.hard_pass", 1.0 if g["hard_pass"] else 0.0, comment="; ".join(g["hard"]))
    obs.score("metric.lra_lu", float(a["ffmpeg"]["LRA_lu"] or 0), comment=f"v{nxt}")
    obs.score("metric.limiter_gr_db", float(a["render"]["limiter_max_gr_db"]), comment=f"v{nxt}")
    if not g["hard_pass"]:
        tries = s.get("fix_attempts", 0) + 1
        if tries > 2:
            return {"stop_reason": f"v{nxt} failed hard gates twice: {g['hard']}"}
        # the failed render is kept under v<next>/attempts/ by the next render
        return {"fix_request": f"v{nxt} rendered but failed hard gates: {g['hard']}. "
                               f"Fix these in the rev >= {nxt} code without losing the musical "
                               "changes.", "fix_attempts": tries, "rev": s["rev"]}
    hist = list(s.get("history", []))
    hist.append({"rev": nxt, "critique_summary": s["critique"]["summary"],
                 "changes": [c["change"] for c in s["critique"]["changes"]],
                 "composer": s["compose"]["summary"][:800], "soft_gates": g["soft"]})
    it = s.get("iteration", 0) + 1
    out = {"prev_rev": s["rev"], "rev": nxt, "iteration": it, "history": hist,
           "fix_attempts": 0, "human_notes": "", "fresh": False}
    if it >= s["max_iters"]:
        out["stop_reason"] = f"reached {s['max_iters']} iteration(s); final revision v{nxt}"
    elif s.get("cost_usd", 0.0) >= s.get("budget_usd", 1e9):
        out["stop_reason"] = f"budget reached (${s['cost_usd']:.2f}); final revision v{nxt}"
    return out


def r_after_render(s: S):
    if s.get("fix_request"):
        return "compose_fix"
    return "listen"           # the final revision is always heard and scored once


def r_after_listen(s: S):
    # in human mode even the final revision pauses once for Jonathan's rating
    if s["mode"] == "human":
        return "human"
    return END if s.get("stop_reason") else "human"


def n_compose_fix(s: S):
    # same composer, now with the fix request; the revision number does not advance
    return n_compose(s)


def build_graph(checkpointer):
    g = StateGraph(S)
    g.add_node("listen", n_listen)
    g.add_node("human", n_human)
    g.add_node("critique", n_critique)
    g.add_node("compose", n_compose)
    g.add_node("compose_fix", n_compose_fix)
    g.add_node("render", n_render)
    g.add_edge(START, "listen")
    g.add_conditional_edges("listen", r_after_listen, ["human", END])
    g.add_conditional_edges("human", r_after_human, ["critique", END])
    g.add_conditional_edges("critique", r_after_critique, ["compose", END])
    g.add_conditional_edges("compose", r_after_compose, ["render", END])
    g.add_conditional_edges("compose_fix", r_after_compose, ["render", END])
    g.add_conditional_edges("render", r_after_render, ["compose_fix", "listen"])
    return g.compile(checkpointer=checkpointer)
