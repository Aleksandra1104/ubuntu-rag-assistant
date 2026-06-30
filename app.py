"""
Streamlit chat UI for the Ask Ubuntu RAG project.

Run it (NOT with `python`) from your project folder, venv active:
    streamlit run app.py

Opens at http://localhost:8501. Shared logic lives in rag_core.py.
Prereqs: pip install streamlit  (plus chromadb, sentence-transformers,
langchain-ollama). Ollama must be running; index built by prepare_dataset.py.
"""

import streamlit as st
from rag_core import (
    get_collection, get_llm, retrieve, is_relevant, build_prompt,
    best_distance, NO_ANSWER, RELEVANCE_THRESHOLD,
)

st.set_page_config(page_title="Ask Ubuntu RAG", page_icon="🐧", layout="centered")


# heavy objects: built once, reused across Streamlit reruns
@st.cache_resource
def load_collection():
    return get_collection()


@st.cache_resource
def load_llm(model_name):
    return get_llm(model_name)


def render_sources(hits, threshold):
    with st.expander(f"📚 {len(hits)} retrieved sources"):
        for text, meta, dist in hits:
            url = meta.get("url", "")
            weak = " · ⚠️ weak match" if dist > threshold else ""
            st.markdown(
                f"**[{url or 'source'}]({url or '#'})** · "
                f"distance `{dist:.3f}` · similarity `{1 - dist:.3f}`{weak}"
            )
            st.caption(text.strip().replace("\n", " ")[:600] + "...")
            st.divider()


# sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="qwen3:8b")
    k = st.slider("Chunks to retrieve", 1, 8, 4)
    threshold = st.slider(
        "Relevance cutoff (cosine distance)", 0.0, 1.0, RELEVANCE_THRESHOLD, 0.05,
        help="If the closest hit is farther than this, refuse to answer.",
    )
    show_sources = st.checkbox("Show retrieved sources", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# open the index
try:
    col = load_collection()
except Exception:
    st.error(
        "Couldn't open the Chroma index. Run `python prepare_dataset.py` first "
        "to build ./chroma_askubuntu, then reload this page."
    )
    st.stop()

llm = load_llm(model_name)

st.title("🐧 Ask Ubuntu RAG")
st.caption(f"Local retrieval over {col.count():,} Ask Ubuntu Q&A · answers by {model_name}")

# history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits") and show_sources:
            render_sources(msg["hits"], threshold)

# new question
if question := st.chat_input("Ask an Ubuntu question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            hits = retrieve(col, question, k)

        if not is_relevant(hits, threshold):
            answer = NO_ANSWER
            st.markdown(answer)
            st.caption(
                f"Closest match was {best_distance(hits):.3f}, beyond the "
                f"{threshold:.2f} cutoff — so I didn't generate an answer."
            )
        else:
            box = st.empty()
            answer = ""
            for chunk in llm.stream(build_prompt(question, hits)):
                answer += chunk.content
                box.markdown(answer)

        if show_sources:
            render_sources(hits, threshold)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )