from typing import List, Dict
import os
from config.settings import settings
from groq import Groq

client = Groq(api_key=settings.GROQ_API_KEY)


def _format_context(chunks: List[Dict]) -> str:
    lines=[]
    for h in chunks:
        cite = f"[{h['source']}{f'p.{h['page']}' if h.get('page') else ''}]"
        lines.append(f"{cite} {h['text']}")
    
    return "\n\n".join(lines)

def _stitched_fallback(question: str, chunks: List[Dict]) -> str:
    #readable fallback if no LLM is Configured

    parts = []
    for h in chunks:
        cite = f"[{h['source']}{f' p.{h['page']}' if h.get('page') else ''}]"
        parts.append(f"{h['text']}\n{cite}")
    body = "\n\n---\n\n".join(parts) if parts else "No context found."
    return f"(context-only draft)\nQ: {question}\n\n{body}"

def generate_answer(question: str, chunks: list[dict]) -> str:
    provider = settings.LLM_PROVIDER.lower()
    print(f"[LLM] provider={provider} model={settings.LLM_MODEL} key_set={bool(settings.GROQ_API_KEY)}")

    if provider == "none":
        print("[LLM] provider disabled -> stitched fallback")
        return _stitched_fallback(question, chunks)

    ctx = _format_context(chunks)
    system = ("You are a legal research assistant (not a lawyer). "
              "Answer STRICTLY from the provided context, with inline citations like [doc p.X]. "
              "If context is insufficient, say so.")
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nWrite a 3-6 sentence answer with citations."

    if provider == "groq":
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_completion_tokens=settings.MAX_TOKENS,
                temperature=0.2,
            )
            
            out = resp.choices[0].message.content.strip()
            print("[LLM] Groq OK; usage:", getattr(resp, "usage", None))
            return out
        except Exception as e:
            print("[LLM] Groq error -> fallback:", repr(e))
            return _stitched_fallback(question, chunks)

    print("[LLM] unknown provider -> stitched fallback")
    return _stitched_fallback(question, chunks)
