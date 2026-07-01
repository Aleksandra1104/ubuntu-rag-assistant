"""
prepare_dataset_aws.py — AWS counterpart of prepare_dataset.py.

Builds the SAME Ask Ubuntu documents and the SAME all-MiniLM-L6-v2 embeddings
as the local Chroma version, but stores them in Amazon S3 Vectors instead of
ChromaDB. Keeping the documents and embedding model identical is what makes the
local-vs-cloud retrieval comparison fair — only the vector store changes.

Run once, after `aws configure` and with Bedrock/S3 Vectors available in us-east-1:
    python prepare_dataset_aws.py

Prereqs:
    pip install "boto3>=1.42" sentence-transformers pandas beautifulsoup4
"""

import html
import boto3
import pandas as pd
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer

# --- config (must match the local build for a clean comparison) ---
REGION = "us-east-1"
BUCKET = "ubuntu-rag-vectors"       # vector bucket name (created if missing)
INDEX = "askubuntu"                 # vector index name
EMBED_MODEL = "all-MiniLM-L6-v2"    # same model as the local Chroma build
DIM = 384                           # all-MiniLM-L6-v2 output dimension
DISTANCE = "cosine"                 # match the local Chroma cosine space
CSV_PATH = "QueryResults.csv"
UPLOAD_BATCH = 500                  # S3 Vectors put_vectors max per call


def clean_html(raw) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace (same as local)."""
    if not isinstance(raw, str):
        return ""
    if "<" in raw:
        raw = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    return " ".join(html.unescape(raw).split())


def row_to_document(row) -> str:
    """
    Fold one CSV row into a single text block.

    """
    title = clean_html(row.get("Title"))
    question = clean_html(row.get("QuestionBody"))
    answer = clean_html(row.get("AcceptedAnswer"))
    return f"Title: {title}\n\nQuestion: {question}\n\nAccepted answer: {answer}"


def build_records(df):
    """Return parallel lists of ids, documents, and metadata dicts."""
    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        title = clean_html(row.get("Title"))
        answer = clean_html(row.get("AcceptedAnswer"))
        if not title or not answer:        # skip rows missing title or answer
            continue
        qid = str(row["QuestionId"])
        ids.append(qid)
        docs.append(row_to_document(row))
        metas.append({
            "text": row_to_document(row),                       # non-filterable: the chunk text (S3 Vectors needs it in metadata)
            "question_id": int(qid),                            # same fields as the local Chroma build
            "tags": str(row.get("Tags", "")),                  # raw, like the local build (tags look like <apt><gnome>)
            "question_score": int(row.get("QuestionScore", 0) or 0),
            "answer_score": int(row.get("AnswerScore", 0) or 0),
            "url": f"https://askubuntu.com/q/{qid}",
        })
    return ids, docs, metas


def ensure_bucket_and_index(s3v):
    """Create the vector bucket and index if they don't already exist."""
    try:
        s3v.create_vector_bucket(vectorBucketName=BUCKET)
        print(f"Created vector bucket: {BUCKET}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"Vector bucket already exists: {BUCKET}")
        else:
            raise

    try:
        s3v.create_index(
            vectorBucketName=BUCKET,
            indexName=INDEX,
            dataType="float32",
            dimension=DIM,
            distanceMetric=DISTANCE,
            # the long document text is stored but never filtered on
            metadataConfiguration={"nonFilterableMetadataKeys": ["text"]},
        )
        print(f"Created index: {INDEX} (dim={DIM}, {DISTANCE})")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"Index already exists: {INDEX}")
        else:
            raise


def main():
    print("Reading CSV and building documents...")
    df = pd.read_csv(CSV_PATH)
    ids, docs, metas = build_records(df)
    print(f"Built {len(docs)} documents.")

    print(f"Loading embedding model {EMBED_MODEL} and embedding (CPU, takes a few minutes)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)

    s3v = boto3.client("s3vectors", region_name=REGION)
    ensure_bucket_and_index(s3v)

    print(f"Uploading {len(ids)} vectors in batches of {UPLOAD_BATCH}...")
    for i in range(0, len(ids), UPLOAD_BATCH):
        batch = [
            {
                "key": ids[j],
                "data": {"float32": embeddings[j].tolist()},
                "metadata": metas[j],
            }
            for j in range(i, min(i + UPLOAD_BATCH, len(ids)))
        ]
        s3v.put_vectors(vectorBucketName=BUCKET, indexName=INDEX, vectors=batch)
        print(f"  uploaded {min(i + UPLOAD_BATCH, len(ids))}/{len(ids)}")

    print("Done. Vectors are in S3 Vectors and ready to query.")


if __name__ == "__main__":
    main()