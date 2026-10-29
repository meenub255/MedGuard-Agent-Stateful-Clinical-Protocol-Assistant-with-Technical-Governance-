# 🛡️ MedGuard — Stateful Clinical Protocol Assistant with Technical Governance

A LangGraph agent that never returns raw LLM output. Every answer is retrieved from a Chroma
knowledge base of clinical guidance, drafted, then **validated for groundedness and mandatory
medical disclaimers** before release. Failures loop back to rewrite, fall back to web search,
or end in a safe refusal.

## Architecture

```mermaid
flowchart TD
    Q([User question]) --> R[Retriever node<br/>Chroma top-k + relevance floor]
    R --> D[Drafting node<br/>LLM or extractive drafter]
    D --> G{Governance / Evaluator node<br/>1. Groundedness ≥ 0.8<br/>2. Disclaimer tag present}
    G -- pass --> O([Output: answer + sources])
    G -- fail, has context,<br/>attempts left --> W[Rewrite node<br/>add disclaimer / strip unsupported claims]
    W --> G
    G -- fail, no context --> S[Web search fallback<br/>Tavily, once]
    S -- results --> D
    S -- no results --> X
    G -- attempts exhausted --> X([Safe refusal + disclaimer])
```

### Governance decision logic

```mermaid
flowchart LR
    A[Draft + context] --> B{Context empty?}
    B -- yes --> F[FAIL: ungrounded]
    B -- no --> C[Score each sentence:<br/>share of content words found in context]
    C --> E{≥ 80% of sentences<br/>supported?}
    E -- no --> F
    E -- yes --> H{Contains<br/>MEDICAL DISCLAIMER tag?}
    F --> H
    H -- no --> I[FAIL: missing disclaimer]
    H -- yes --> J{Any failures?}
    I --> J
    J -- none --> K[PASS → output]
    J -- some --> L[Route: rewrite / web search / refusal]
```

### Shared graph state

```mermaid
classDiagram
    class State {
      question: str
      context: list[source, text]
      draft: str
      issues: list[str]
      attempts: int
      used_web: bool
      final: str
      trace: list[str]
    }
```

## UI walkthrough

### 1. Home screen
![Home](docs/images/01_home.png)

1. **Optional API keys** – OpenAI (LLM drafting) and Tavily (web fallback); kept in session memory only.
2. **Governance rules** – the gates every answer must clear.
3. **Example questions** – one click fills the question box (includes an out-of-scope probe).
4. **Question box** – type any clinical question and press *Ask*.

### 2. Grounded answer (governance passes)
![Grounded answer](docs/images/02_grounded_answer.png)

1. **Status banner** – green: groundedness and disclaimer checks passed.
2. **Live agent trace** – each LangGraph node as it runs (retrieve → draft → govern → output).
3. **Answer** – cited sentences from the guideline plus the mandatory `[MEDICAL DISCLAIMER]` tag.
4. **Sources** – the exact retrieved passages, expandable for audit.

### 3. Safe refusal (out of scope)
![Safe refusal](docs/images/03_safe_refusal.png)

1. **Red banner** – no guideline matched, so nothing is fabricated.
2. **Trace** – retrieve found nothing → governance failed → web fallback attempted → refusal.
3. **Answer** – a refusal that still carries the disclaimer.

## Quick start

```bash
pip install langgraph langchain-openai langchain-tavily chromadb streamlit
streamlit run app.py          # web UI
python medguard.py "How often should type 2 diabetics get eye screening?"   # CLI
```

Optional environment variables: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MEDGUARD_MODEL` (default `gpt-4o-mini`).
Without keys it runs offline with a deterministic extractive drafter.

Regenerate screenshots: `pip install playwright pillow && python docs/make_screenshots.py`

## Files

| File | Purpose |
|---|---|
| `medguard.py` | Graph, nodes, governance and routing |
| `guidelines.py` | Demo guideline corpus (replace with vetted protocols) |
| `app.py` | Streamlit UI |
| `docs/make_screenshots.py` | Captures and annotates UI screenshots |

> ⚠️ Informational only; not medical advice. The bundled corpus is a paraphrased demo.
