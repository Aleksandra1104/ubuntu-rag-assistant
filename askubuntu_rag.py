"""
Ask Ubuntu RAG — retrieve relevant Q&A from your Chroma index, then have a
local Ollama model write an answer grounded in what was retrieved.

All the real logic lives in rag_core.py; this just calls it and prints.

    python askubuntu_rag.py
"""

from rag_core import get_collection, get_llm, answer, RELEVANCE_THRESHOLD

if __name__ == "__main__":
    col = get_collection()
    llm = get_llm("qwen3:8b")

    question = "How do I list all installed packages?"
    response, hits, used_kb = answer(col, llm, question)

    print(f"Question: {question}\n")
    print("Retrieved (closest first — lower cosine distance = more similar):")
    for text, meta, dist in hits:
        weak = "  (weak)" if dist > RELEVANCE_THRESHOLD else ""
        print(f"  dist={dist:.3f}  sim={1 - dist:.3f}  {meta.get('url', '')}{weak}")
        print(f"    {text[:110].strip().replace(chr(10), ' ')}...")

    tag = "answered from KB" if used_kb else "relevance gate tripped — refused"
    print(f"\nAnswer ({tag}):\n", response)