"""Check each service authenticates, without printing any secret.

    python -m agent.check
"""
import os

from .config import CLAUDE_MODEL, GEMINI_MODEL, load_env


def main():
    load_env()
    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            m = anthropic.Anthropic().models.retrieve(CLAUDE_MODEL)
            print(f"anthropic: ok ({m.id})")
        except Exception as e:
            print(f"anthropic: FAILED ({type(e).__name__})")
    else:
        print("anthropic: no ANTHROPIC_API_KEY in instrumentals/.env")
    # Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            gc = genai.Client(api_key=os.environ["GEMINI_API_KEY"])   # keep a reference while paging
            names = [m.name for m in gc.models.list()
                     if "generateContent" in (m.supported_actions or [])]
            gem = sorted(n.split("/")[-1] for n in names if "gemini" in n)
            print(f"gemini: ok; configured model {GEMINI_MODEL} "
                  f"{'is' if GEMINI_MODEL in gem else 'is NOT'} available")
            print("  models that generate content:", ", ".join(g for g in gem
                                                              if "pro" in g or "flash" in g))
        except Exception as e:
            print(f"gemini: FAILED ({type(e).__name__})")
    else:
        print("gemini: no GEMINI_API_KEY in instrumentals/.env")
    # Langfuse
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client
            print(f"langfuse: {'ok' if get_client().auth_check() else 'FAILED auth'}")
        except Exception as e:
            print(f"langfuse: FAILED ({type(e).__name__})")
    else:
        print("langfuse: no keys (tracing disabled; everything still runs)")


if __name__ == "__main__":
    main()
