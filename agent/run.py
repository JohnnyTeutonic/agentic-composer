"""Run the music agent.

Start (human mode pauses after each listen for Jonathan's rating and notes):
    python -m agent.run start a --iters 2 --mode human --budget 10
    python -m agent.run start c --iters 3 --mode auto  --budget 15
    python -m agent.run start c --fresh --iters 1 --mode auto --budget 8 --notes "start over"

Resume a paused run:
    python -m agent.run resume <thread> --rating 8 --notes "the bridge drags; more of the 2/4 jolt"
    python -m agent.run resume <thread> --rating fire --stop

Inspect:
    python -m agent.run status <thread>
"""
import argparse
import json
import sqlite3
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import obs
from .config import STATE_DB, PIECES, load_env, have
from .graph import build_graph
from .tools import latest_rev


def _graph(conn):
    return build_graph(SqliteSaver(conn))


def _report(result, thread):
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(json.dumps(payload, indent=1))
        print(f"\nPaused for you. Listen to {payload['wav']}, then:\n"
              f"  python -m agent.run resume {thread} --rating <score> --notes \"...\"  "
              f"[--stop]")
    else:
        print(json.dumps({"stopped": result.get("stop_reason"), "rev": result.get("rev"),
                          "cost_usd": round(result.get("cost_usd", 0.0), 2),
                          "history": result.get("history", [])}, indent=1))


def _invoke(graph, payload, thread, key):
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 60}
    with obs.session(thread, key):
        with obs.span(f"music-agent {key}", as_type="agent", input={"thread": thread}) as root:
            result = graph.invoke(payload, cfg)
            if root:
                root.update(output={"stop": result.get("stop_reason"), "rev": result.get("rev")})
    obs.flush()
    return result


def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("piece", choices=list(PIECES))
    st.add_argument("--from-rev", type=int)
    st.add_argument("--iters", type=int, default=2)
    st.add_argument("--mode", choices=["human", "auto"], default="human")
    st.add_argument("--budget", type=float, default=10.0, help="USD ceiling on Claude spend")
    st.add_argument("--thread")
    st.add_argument("--notes", default="", help="Jonathan's notes on the starting revision")
    st.add_argument("--fresh", action="store_true",
                    help="the first round writes a new composition instead of revising")
    rs = sub.add_parser("resume")
    rs.add_argument("thread")
    rs.add_argument("--rating")
    rs.add_argument("--notes", default="")
    rs.add_argument("--stop", action="store_true")
    ss = sub.add_parser("status")
    ss.add_argument("thread")
    a = ap.parse_args()

    if a.cmd in ("start", "resume") and not have("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY and GEMINI_API_KEY are needed (instrumentals/.env); "
                 "run `python -m agent.check`.")
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STATE_DB), check_same_thread=False)
    graph = _graph(conn)

    if a.cmd == "start":
        rev = a.from_rev or latest_rev(a.piece)
        thread = a.thread or f"{a.piece}-v{rev}-{a.mode}"
        n = 1
        while graph.get_state({"configurable": {"thread_id": thread}}).values and n < 100:
            n += 1
            thread = f"{a.piece}-v{rev}-{a.mode}-{n}"
        init = {"key": a.piece, "mode": a.mode, "rev": rev, "prev_rev": rev - 1 if rev > 1 else 0,
                "max_iters": a.iters, "iteration": 0, "budget_usd": a.budget, "cost_usd": 0.0,
                "listening": {}, "history": [], "human_notes": a.notes,
                "fresh": a.fresh}
        print(f"thread {thread}: {PIECES[a.piece]['title']} from v{rev}, mode {a.mode}")
        _report(_invoke(graph, init, thread, a.piece), thread)
    elif a.cmd == "resume":
        state = graph.get_state({"configurable": {"thread_id": a.thread}}).values
        if not state:
            sys.exit(f"no such thread {a.thread}")
        rating = a.rating
        try:
            rating = float(rating) if rating is not None else None
        except ValueError:
            pass
        _report(_invoke(graph, Command(resume={"rating": rating, "notes": a.notes,
                                               "stop": a.stop}), a.thread, state["key"]),
                a.thread)
    else:
        snap = graph.get_state({"configurable": {"thread_id": a.thread}})
        v = snap.values
        print(json.dumps({"next": list(snap.next), "rev": v.get("rev"),
                          "iteration": v.get("iteration"), "cost_usd": v.get("cost_usd"),
                          "stop": v.get("stop_reason"), "history": v.get("history")}, indent=1))


if __name__ == "__main__":
    main()
