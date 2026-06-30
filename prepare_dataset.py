"""
Prepare an Ask Ubuntu SEDE export for embedding.

Reads the CSV downloaded from data.stackexchange.com, strips the HTML from
the question and answer bodies, and builds one clean text document per row,
plus parallel metadata and ids.

The returned (documents, metadatas, ids) lists drop straight into a Chroma
collection's .upsert(documents=..., metadatas=..., ids=...) call.
"""

import html
import pandas as pd
from bs4 import BeautifulSoup


def clean_html(raw) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace."""
    if not isinstance(raw, str):
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    return " ".join(html.unescape(text).split())


def build_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    documents, metadatas, ids = [], [], []

    for _, row in df.iterrows():
        title = clean_html(row.get("Title", ""))
        question = clean_html(row.get("QuestionBody", ""))
        answer = clean_html(row.get("AcceptedAnswer", ""))

        # skip rows missing the essentials
        if not title or not answer:
            continue

        qid = int(row["QuestionId"])
        documents.append(
            f"Title: {title}\n\nQuestion: {question}\n\nAccepted answer: {answer}"
        )
        metadatas.append({
            "question_id": qid,
            "tags": str(row.get("Tags", "")),
            "question_score": int(row.get("QuestionScore", 0) or 0),
            "answer_score": int(row.get("AnswerScore", 0) or 0),
            "url": f"https://askubuntu.com/q/{qid}",
        })
        ids.append(str(qid))

    return documents, metadatas, ids


if __name__ == "__main__":

    docs, metas, ids = build_documents("QueryResults.csv")

    print(f"Built {len(docs)} clean documents\n" + "-" * 60)
    print(docs[0][:500] + ("..." if len(docs[0]) > 500 else ""))
    print("-" * 60)
    print(metas[0])

    # Next step (your existing stack):
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_askubuntu")
    
    col = client.get_or_create_collection(
        name="askubuntu",
        embedding_function=ef,
        configuration={"hnsw": {"space": "cosine"}},  # The default L2 is built for magnitude, but sentence-transformers models like all-MiniLM-L6-v2 produce normalized embeddings tuned for angle-based comparison, so cosine is the metric they were trained for.
    )

    B = 5000  # under Chroma's per-call max of 5461
    for i in range(0, len(docs), B):
        col.upsert(
            documents=docs[i:i+B],
            metadatas=metas[i:i+B],
            ids=ids[i:i+B],
        )
        print(f"added {min(i+B, len(docs))}/{len(docs)}")