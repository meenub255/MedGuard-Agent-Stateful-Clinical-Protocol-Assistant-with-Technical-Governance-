"""Streamlit UI for MedGuard. Run: streamlit run app.py"""
import os
import streamlit as st

st.set_page_config(page_title="MedGuard", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    key = st.text_input("OpenAI API key (optional)", type="password",
                        help="Kept in memory for this session only. Without it, an offline extractive drafter is used.")
    if key: os.environ["OPENAI_API_KEY"] = key
    tav = st.text_input("Tavily API key (web fallback, optional)", type="password")
    if tav: os.environ["TAVILY_API_KEY"] = tav
    st.caption(f"Drafter: **{'OpenAI LLM' if os.getenv('OPENAI_API_KEY') else 'Offline extractive'}**  \n"
               f"Web fallback: **{'Tavily' if os.getenv('TAVILY_API_KEY') else 'disabled'}**")
    st.divider()
    st.markdown("**Governance gates**\n- ✅ Groundedness ≥ 0.8\n- ✅ `[MEDICAL DISCLAIMER]` tag\n- 🔁 Rewrite ≤ 2×\n- 🌐 Web fallback\n- 🚫 Safe refusal")

import medguard as mg  # imported after env is set up

st.title("🛡️ MedGuard")
st.caption("Governed clinical-guidance agent · LangGraph + Chroma · every answer is validated before release")

EXAMPLES = ["How often should type 2 diabetes patients get eye screening?",
            "When should a patient be referred urgently to an ophthalmologist?",
            "Can St John's wort be taken with antidepressants?",
            "What is the best treatment for a broken leg?"]
cols = st.columns(len(EXAMPLES))
for c, e in zip(cols, EXAMPLES):
    if c.button(e, use_container_width=True): st.session_state["q"] = e

q = st.text_input("Ask a clinical question", key="q", placeholder="e.g. Which herbs interact with anticoagulants?")
go = st.button("Ask", type="primary")

ICON = {"retrieve": "🔎", "draft": "✍️", "govern": "⚖️", "rewrite": "🔁",
        "web_search": "🌐", "output": "✅", "safe_refusal": "🚫"}

if go and q.strip():
    left, right = st.columns([3, 2])
    steps, state = [], {}
    with right:
        st.subheader("Agent trace")
        box = st.container()
    with left:
        with st.spinner("Running governed pipeline..."):
            for upd in mg.APP.stream({"question": q}, stream_mode="updates"):
                for node, delta in upd.items():
                    state.update(delta)
                    msg = delta.get("trace", [""])[-1]
                    steps.append((node, msg, delta))
                    box.markdown(f"{ICON.get(node, '•')} **{node}** — {msg.split(': ', 1)[-1]}")
    with left:
        refused = any(n == "safe_refusal" for n, _, _ in steps)
        (st.error if refused else st.success)("Safe refusal — no grounded answer available" if refused
                                              else "Passed governance checks")
        st.subheader("Answer")
        st.markdown(state.get("final", ""))
        if state.get("used_web") and not refused: st.warning("Answer used web search fallback — verify independently.")
        st.subheader("Sources")
        for c in state.get("context", []) or [{"text": "No sources retrieved.", "source": "—"}]:
            with st.expander(c["source"]): st.write(c["text"])
        with st.expander("Validation issues from last governance pass"):
            gov = [d for n, _, d in steps if n == "govern"]
            for i, d in enumerate(gov, 1):
                st.write(f"Pass {i}: {d.get('issues') or '✅ none'}")
elif go:
    st.info("Enter a question first.")

st.divider()
st.caption("⚠️ Informational only; not medical advice. Demo guideline corpus — replace with vetted protocols.")
