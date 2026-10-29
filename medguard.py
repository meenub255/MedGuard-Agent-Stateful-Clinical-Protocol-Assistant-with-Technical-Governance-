"""MedGuard: governed clinical-guidance agent (LangGraph + Chroma).

Flow:  retrieve -> draft -> govern --pass--> output
                              |--fail (attempts left)--> rewrite -> govern
                              |--fail (ungrounded/no context)--> web_search -> draft
                              |--fail (exhausted)--> safe_refusal
Runs offline (extractive drafter, hash embeddings) or with OPENAI_API_KEY / TAVILY_API_KEY.
"""
import hashlib, math, os, re, sys
from typing import TypedDict, List, Literal

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langgraph.graph import StateGraph, START, END

from guidelines import GUIDELINES

DISCLAIMER = "[MEDICAL DISCLAIMER] This is informational only and not a substitute for professional medical advice, diagnosis or treatment. Consult a qualified healthcare provider."
DISCLAIMER_TAG = "[MEDICAL DISCLAIMER]"
TOP_K = 3
MAX_REWRITES = 2
MIN_RELEVANCE = 0.15          # cosine similarity floor for retrieved chunks
GROUNDEDNESS_THRESHOLD = 0.8  # fraction of answer sentences supported by context

STOP = set("the a an and or of to in for with is are be may can should it its on as by at from that this".split())
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


class HashEmbedding(EmbeddingFunction):
    """Deterministic offline bag-of-words embedding (swap for a real model in production)."""
    DIM = 512
    def __init__(self): pass
    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for t in input:
            v = [0.0] * self.DIM
            for w in tok(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
    @staticmethod
    def name(): return "hash-embedding"


def build_store():
    col = chromadb.EphemeralClient().get_or_create_collection(
        "guidelines", embedding_function=HashEmbedding(), metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        col.add(ids=[g["id"] for g in GUIDELINES], documents=[g["text"] for g in GUIDELINES],
                metadatas=[{"source": g["source"]} for g in GUIDELINES])
    return col

STORE = build_store()


class State(TypedDict, total=False):
    question: str
    context: List[dict]      # {text, source}
    draft: str
    issues: List[str]
    attempts: int
    used_web: bool
    final: str
    trace: List[str]


def log(s, msg): return s.get("trace", []) + [msg]

# ---------- LLM (optional) ----------
def llm(prompt: str):
    if not os.getenv("OPENAI_API_KEY"): return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("MEDGUARD_MODEL", "gpt-4o-mini"), temperature=0).invoke(prompt).content

def sentences(t): return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]

# ---------- Nodes ----------
def retrieve(s: State) -> State:
    r = STORE.query(query_texts=[s["question"]], n_results=TOP_K)
    ctx = [{"text": d, "source": m["source"]}
           for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
           if 1 - dist >= MIN_RELEVANCE]
    return {"context": ctx, "attempts": 0, "used_web": False,
            "trace": log(s, f"retrieve: {len(ctx)} relevant chunk(s)")}


def draft(s: State) -> State:
    ctx = s["context"]
    if not ctx:
        return {"draft": "", "trace": log(s, "draft: no context")}
    ctx_txt = "\n".join(f"[{c['source']}] {c['text']}" for c in ctx)
    out = llm(f"Answer ONLY using the context. Cite sources in brackets. End with the exact line: {DISCLAIMER}\n\nContext:\n{ctx_txt}\n\nQuestion: {s['question']}")
    if out is None:  # offline extractive drafter: pick best-matching sentences
        q = set(tok(s["question"]))
        cand = [(len(q & set(tok(x))), x, c["source"]) for c in ctx for x in sentences(c["text"])]
        top = [f"{x} [{src}]" for sc, x, src in sorted(cand, key=lambda t: -t[0])[:3] if sc]
        out = " ".join(top) + "\n\n" + DISCLAIMER
    return {"draft": out, "trace": log(s, "draft: generated")}


def evaluate(draft_text: str, ctx: List[dict]):
    issues = []
    body = draft_text.replace(DISCLAIMER, "").replace(DISCLAIMER_TAG, "")
    ctx_vocab = set(w for c in ctx for w in tok(c["text"]))
    sents = [x for x in sentences(re.sub(r"\[[^\]]*\]", "", body)) if tok(x)]
    scores = [len(set(tok(x)) & ctx_vocab) / len(set(tok(x))) for x in sents]
    ground = sum(sc >= 0.85 for sc in scores) / len(scores) if scores else 0.0
    if not ctx or ground < GROUNDEDNESS_THRESHOLD:
        issues.append(f"ungrounded (score {ground:.2f} < {GROUNDEDNESS_THRESHOLD})")
    if DISCLAIMER_TAG not in draft_text:
        issues.append("missing disclaimer tag")
    return issues, ground


def govern(s: State) -> State:
    issues, g = evaluate(s.get("draft", ""), s["context"])
    return {"issues": issues, "trace": log(s, f"govern: groundedness={g:.2f} issues={issues or 'none'}")}


def rewrite(s: State) -> State:
    d = s["draft"]
    if any("disclaimer" in i for i in s["issues"]):
        d = d.rstrip() + "\n\n" + DISCLAIMER
    if any("ungrounded" in i for i in s["issues"]):
        ctx_txt = "\n".join(c["text"] for c in s["context"])
        out = llm(f"Rewrite using ONLY the context; remove unsupported claims. Keep line: {DISCLAIMER}\nContext:\n{ctx_txt}\nDraft:\n{d}")
        if out: d = out
        else:  # offline: drop sentences not supported by context
            vocab = set(w for c in s["context"] for w in tok(c["text"]))
            keep = [x for x in sentences(d.replace(DISCLAIMER, ""))
                    if not tok(re.sub(r"\[[^\]]*\]", "", x)) or
                    len(set(tok(re.sub(r"\[[^\]]*\]", "", x))) & vocab) / len(set(tok(re.sub(r"\[[^\]]*\]", "", x)))) >= 0.85]
            d = " ".join(keep) + "\n\n" + DISCLAIMER
    return {"draft": d, "attempts": s["attempts"] + 1, "trace": log(s, "rewrite")}


def web_search(s: State) -> State:
    results = []
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch
        r = TavilySearch(max_results=3).invoke(s["question"])
        results = [{"text": x["content"], "source": x["url"]} for x in r.get("results", [])]
    return {"context": results, "used_web": True, "attempts": s["attempts"] + 1,
            "trace": log(s, f"web_search: {len(results)} result(s)")}


def output(s: State) -> State:
    note = "\n\n(Sources include web search fallback; verify independently.)" if s.get("used_web") else ""
    return {"final": s["draft"] + note, "trace": log(s, "output")}


def safe_refusal(s: State) -> State:
    return {"final": "I could not produce a sufficiently grounded answer from the available guidelines. "
                     "Please consult a qualified healthcare professional.\n\n" + DISCLAIMER,
            "trace": log(s, "safe_refusal")}

# ---------- Routing ----------
def route(s: State) -> Literal["output", "rewrite", "web_search", "safe_refusal"]:
    if not s["issues"]: return "output"
    if s["attempts"] >= MAX_REWRITES + 1: return "safe_refusal"
    ungrounded_no_ctx = not s["context"] or not s.get("draft")
    if ungrounded_no_ctx and not s["used_web"]: return "web_search"
    if ungrounded_no_ctx: return "safe_refusal"
    return "rewrite"

def after_web(s: State) -> Literal["draft", "safe_refusal"]:
    return "draft" if s["context"] else "safe_refusal"


def build_graph():
    g = StateGraph(State)
    for n, f in [("retrieve", retrieve), ("draft", draft), ("govern", govern), ("rewrite", rewrite),
                 ("web_search", web_search), ("output", output), ("safe_refusal", safe_refusal)]:
        g.add_node(n, f)
    g.add_edge(START, "retrieve"); g.add_edge("retrieve", "draft"); g.add_edge("draft", "govern")
    g.add_conditional_edges("govern", route)
    g.add_edge("rewrite", "govern")
    g.add_conditional_edges("web_search", after_web)
    g.add_edge("output", END); g.add_edge("safe_refusal", END)
    return g.compile()

APP = build_graph()

def ask(q: str) -> State:
    return APP.invoke({"question": q})

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How often should type 2 diabetes patients get eye screening?"
    r = ask(q)
    print("\n".join(" -> " + t for t in r["trace"])); print("\n" + r["final"])

