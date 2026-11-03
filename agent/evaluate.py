"""Langfuse evaluation harness: does the automated listener agree with Jonathan's ear?

    python -m agent.evaluate rate a 3 9 --notes "the hook locks with the bass"   # your rating
    python -m agent.evaluate sync                                                # renders -> dataset
    python -m agent.evaluate run --name listener-baseline                        # experiment run
    python -m agent.evaluate run --name flash-test --model gemini-2.5-flash      # compare models

Dataset "instrumentals-renders": one item per rendered revision. Its input is the piece and
revision, and its expected output is Jonathan's rating where he has given one. The experiment task
is the Gemini listener. Item evaluators record the technical gates, the listener's scores and,
for rated items, the listener's error against Jonathan's rating. A run evaluator reports
agreement over all rated items. Comparing runs in the Langfuse UI shows which listener set-up
tracks his ear best.
"""
import argparse
import os

from . import config
from .config import PIECES, load_env
from .tools import read_analysis, gates, ratings, add_rating, PROJ

DATASET = "instrumentals-renders"


def _renders():
    for key, p in PIECES.items():
        base = PROJ / "out" / p["slug"]
        for d in sorted(base.glob("v*")):
            if d.name[1:].isdigit() and (d / f"{p['slug']}_{d.name}.wav").exists():
                yield key, int(d.name[1:])


def _numeric(r):
    try:
        return float(r["score"])
    except (TypeError, ValueError, KeyError):
        return None


def sync(lf):
    lf.create_dataset(name=DATASET, description="Every rendered revision of the instrumentals; "
                      "expected output is Jonathan's rating where given.")
    n = 0
    for key, rev in _renders():
        r = ratings(key).get(str(rev))
        lf.create_dataset_item(
            dataset_name=DATASET, id=f"{PIECES[key]['slug']}-v{rev}",
            input={"piece": key, "rev": rev, "title": PIECES[key]["title"],
                   "brief": PIECES[key]["brief"]},
            expected_output={"jonathan_rating": r["score"], "jonathan_note": r["note"],
                             "rater": r.get("rater", "Jonathan")}
            if r else None,
            metadata={"slug": PIECES[key]["slug"]})
        n += 1
    print(f"synced {n} renders into dataset {DATASET}")


def _inp(item):
    return item.input if hasattr(item, "input") else item["input"]


def task(*, item, **_):
    from .listen import listen
    i = _inp(item)
    return listen(i["piece"], i["rev"])


def ev_gates(*, input, output, expected_output, metadata, **_):
    from langfuse import Evaluation
    a = read_analysis(input["piece"], input["rev"])
    g = gates(input["piece"], a)
    return [Evaluation(name="gates.hard_pass", value=1.0 if g["hard_pass"] else 0.0,
                       comment="; ".join(g["hard"] + g["soft"]) or "all gates pass"),
            Evaluation(name="metric.lra_lu", value=float(a["ffmpeg"]["LRA_lu"] or 0)),
            Evaluation(name="metric.limiter_gr_db", value=float(a["render"]["limiter_max_gr_db"])),
            Evaluation(name="metric.section_contrast", value=float(a.get("section_contrast") or 0))]


def ev_listener(*, input, output, expected_output, metadata, **_):
    from langfuse import Evaluation
    return [Evaluation(name=f"listener.{k}", value=float(v)) for k, v in output["scores"].items()]


def ev_agreement(*, input, output, expected_output, metadata, **_):
    from langfuse import Evaluation
    if not expected_output:
        return []
    j = _numeric({"score": expected_output.get("jonathan_rating")})
    if j is None:
        return [Evaluation(name="jonathan.rating_label", value=str(expected_output["jonathan_rating"]),
                           comment=expected_output.get("jonathan_note"))]
    return [Evaluation(name="listener.abs_error_vs_jonathan",
                       value=abs(float(output["scores"]["overall"]) - j),
                       comment=f"listener {output['scores']['overall']} vs Jonathan {j}")]


def run_agreement(*, item_results, **_):
    from langfuse import Evaluation
    pairs = []
    for r in item_results:
        exp = getattr(r.item, "expected_output", None) if hasattr(r, "item") else None
        j = _numeric({"score": (exp or {}).get("jonathan_rating")}) if exp else None
        if j is not None and r.output:
            pairs.append((float(r.output["scores"]["overall"]), j))
    if len(pairs) < 3:
        return [Evaluation(name="agreement.n_rated", value=float(len(pairs)),
                           comment="need at least 3 numerically rated renders for a correlation")]
    import numpy as np
    from scipy.stats import spearmanr
    x, y = np.array(pairs).T
    rho = spearmanr(x, y).statistic
    return [Evaluation(name="agreement.n_rated", value=float(len(pairs))),
            Evaluation(name="agreement.spearman", value=float(rho)),
            Evaluation(name="agreement.mean_abs_error", value=float(np.mean(np.abs(x - y))))]


def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate")
    r.add_argument("piece", choices=list(PIECES))
    r.add_argument("rev", type=int)
    r.add_argument("score")
    r.add_argument("--notes", default="")
    sub.add_parser("sync")
    e = sub.add_parser("run")
    e.add_argument("--name", required=True)
    e.add_argument("--model", help="Gemini model for this run")
    a = ap.parse_args()

    if a.cmd == "rate":
        add_rating(a.piece, a.rev, a.score, a.notes)
        print(f"recorded {PIECES[a.piece]['title']} v{a.rev}: {a.score}")
        return
    from .obs import client
    lf = client()
    if not lf:
        raise SystemExit("Langfuse keys are needed for sync and run (instrumentals/.env)")
    if a.cmd == "sync":
        sync(lf)
        return
    if a.model:
        os.environ["MUSIC_GEMINI_MODEL"] = a.model
        config.GEMINI_MODEL = a.model
        from . import listen as L
        L.GEMINI_MODEL = a.model
    ds = lf.get_dataset(DATASET)
    res = ds.run_experiment(name=a.name, description=f"listener {config.GEMINI_MODEL}",
                            task=task, evaluators=[ev_gates, ev_listener, ev_agreement],
                            run_evaluators=[run_agreement], max_concurrency=2,
                            metadata={"gemini_model": config.GEMINI_MODEL})
    lf.flush()
    print(res.format() if hasattr(res, "format") else res)


if __name__ == "__main__":
    main()
