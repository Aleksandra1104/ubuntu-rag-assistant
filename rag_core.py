"""
Shared RAG logic for the Ask Ubuntu project.

Both the terminal script (askubuntu_rag.py) and the Streamlit UI (app.py)
import from here, so retrieval, the prompt, and the relevance gate are in
exactly one place. 
"""

import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama

# must match what prepare_dataset.py used to build the index
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_askubuntu"
COLLECTION = "askubuntu"

# Cosine distance above which the system treats even the BEST hit as "not a real match"
# and refuses to answer. Good hits sat around 0.22-0.29.
RELEVANCE_THRESHOLD = 0.45

NO_ANSWER = (
    "I don't have a good answer for that in the Ask Ubuntu knowledge base — "
    "the closest entries weren't a strong enough match to answer reliably."
)


def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION, embedding_function=ef)

# Used previously llama3.2:3b and qwen2.5:7b-instruct
def get_llm(model_name="qwen3:8b"):
    return ChatOllama(model=model_name, temperature=0, reasoning=False)


def retrieve(col, question, k=4):
    """Return the k closest chunks as (text, metadata, distance)."""
    res = col.query(query_texts=[question], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def best_distance(hits):
    """Smallest (closest) cosine distance among the hits, or None if empty."""
    return min((dist for _t, _m, dist in hits), default=None)


def is_relevant(hits, threshold=RELEVANCE_THRESHOLD):
    """True only if the closest hit is near enough to count as a real match."""
    bd = best_distance(hits)
    return bd is not None and bd <= threshold


# def build_prompt(question, hits):
#     context = "\n\n---\n\n".join(text for text, _meta, _dist in hits)
#     return (
#         "You are an Ubuntu support assistant. Use the reference snippets below as "
#         "your only source of facts, but write ONE direct answer to the user's "
#         "question in your own words. Do NOT summarize the snippets one by one or "
#         "refer to them as separate questions. If they don't actually address the "
#         "question, say you don't have a good answer rather than padding.\n\n"
#         f"User question: {question}\n\n"
#         f"Reference snippets:\n{context}\n\nAnswer:"
#     )

# Updated prompt for qwen3:8b to control output style (no emoji, etc.)
def build_prompt(question, hits):
    context = "\n\n---\n\n".join(text for text, _meta, _dist in hits)
    return (
        "You are an Ubuntu support assistant. Use the reference snippets below as "
        "your only source of facts, but write ONE direct answer to the user's "
        "question in your own words. Do NOT summarize the snippets one by one or "
        "refer to them as separate questions. If they don't actually address the "
        "question, say you don't have a good answer rather than padding.\n\n"
        "Style: answer concisely in plain prose. Do not use emoji, markdown "
        "headings, or bulleted lists. Do not add a greeting or a sign-off. Answer "
        "only the question that was asked — do not volunteer tangential or "
        "unrequested information.\n\n"
        f"User question: {question}\n\n"
        f"Reference snippets:\n{context}\n\nAnswer:"
    )


def answer(col, llm, question, k=4, threshold=RELEVANCE_THRESHOLD):
    """
    Full pipeline. Returns (answer_text, hits, used_kb).
    used_kb is False when the relevance gate trips and we refuse to answer.
    """
    hits = retrieve(col, question, k)
    if not is_relevant(hits, threshold):
        return NO_ANSWER, hits, False
    text = llm.invoke(build_prompt(question, hits)).content
    return text, hits, True