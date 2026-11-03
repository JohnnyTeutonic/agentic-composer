"""Langfuse tracing with a silent fallback: without keys every helper is a no-op."""
import contextlib
import os

_client = None


def client():
    """The Langfuse client, or None when keys are absent or authentication fails."""
    global _client
    if _client is None:
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            _client = False
        else:
            from langfuse import get_client
            _client = get_client()
    return _client or None


def span(name, as_type="span", **kw):
    """Context manager for a Langfuse observation (span, generation, agent, tool, evaluator)."""
    c = client()
    if not c:
        return contextlib.nullcontext(None)
    return c.start_as_current_observation(name=name, as_type=as_type, **kw)


def session(session_id, piece, tags=()):
    c = client()
    if not c:
        return contextlib.nullcontext(None)
    from langfuse import propagate_attributes
    return propagate_attributes(session_id=session_id, tags=[f"piece:{piece}", *tags],
                                trace_name=f"music-agent:{piece}")


def score(name, value, comment=None, data_type=None, trace_id=None):
    """Score the current trace (or a given trace). Numeric unless data_type says otherwise."""
    c = client()
    if not c:
        return
    kw = {"name": name, "value": value, "comment": comment}
    if data_type:
        kw["data_type"] = data_type
    if trace_id:
        c.create_score(trace_id=trace_id, **kw)
    else:
        c.score_current_trace(**kw)


def trace_id():
    c = client()
    return c.get_current_trace_id() if c else None


def flush():
    c = client()
    if c:
        c.flush()
