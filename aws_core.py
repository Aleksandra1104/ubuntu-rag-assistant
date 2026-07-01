"""
aws_core.py — AWS counterpart of rag_core.py.

Same pipeline as the local version (retrieve -> relevance gate -> generate),
but retrieval runs against Amazon S3 Vectors and generation runs on Amazon
Bedrock (Claude Haiku 4.5). The query is embedded locally with the SAME
all-MiniLM-L6-v2 model as the index, and the prompt and relevance gate are
imported from rag_core, so the ONLY differences from local are the vector store
and the LLM — which is what makes the comparison fair.

Quick smoke test:
    python aws_core.py

Prereqs: pip install "boto3>=1.42" sentence-transformers
Requires aws configure done, and S3 Vectors + Bedrock available in the region.
"""

import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer

# reuse the exact prompt, gate, and threshold from the local version
from rag_core import build_prompt, is_relevant, best_distance, NO_ANSWER, RELEVANCE_THRESHOLD

REGION = "us-east-1"
BUCKET = "ubuntu-rag-vectors"
INDEX = "askubuntu"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"

# lazily-created singletons (built once, reused)
_embedder = None
_s3vectors = None
_bedrock = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_s3vectors():
    global _s3vectors
    if _s3vectors is None:
        _s3vectors = boto3.client("s3vectors", region_name=REGION)
    return _s3vectors


def get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    return _bedrock


def retrieve(question, k=4):
    """
    Embed the query locally, search S3 Vectors, and return hits as
    (text, metadata, distance) — the same shape rag_core.retrieve returns,
    so build_prompt / is_relevant / the eval harness work unchanged.
    """
    vector = get_embedder().encode([question])[0].tolist()
    resp = get_s3vectors().query_vectors(
        vectorBucketName=BUCKET,
        indexName=INDEX,
        queryVector={"float32": vector},
        topK=k,
        returnDistance=True,
        returnMetadata=True,
    )
    hits = []
    for v in resp.get("vectors", []):
        meta = v.get("metadata", {})
        hits.append((meta.get("text", ""), meta, v.get("distance")))
    return hits


def generate(prompt):
    """Generate an answer with Bedrock (Converse API), with a fallback to the
    cross-region inference profile if the direct model ID isn't accepted."""
    bedrock = get_bedrock()
    kwargs = dict(
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    try:
        resp = bedrock.converse(modelId=MODEL_ID, **kwargs)
    except ClientError as e:
        msg = e.response["Error"].get("Message", "")
        if "inference profile" in msg or "on-demand throughput isn't supported" in msg:
            resp = bedrock.converse(modelId="us." + MODEL_ID, **kwargs)
        else:
            raise
    return resp["output"]["message"]["content"][0]["text"]


def answer(question, k=4, threshold=RELEVANCE_THRESHOLD):
    """
    Full pipeline. Returns (answer_text, hits, used_kb).
    used_kb is False when the relevance gate trips and we refuse to answer.
    Mirrors rag_core.answer so both backends are called the same way.
    """
    hits = retrieve(question, k)
    if not is_relevant(hits, threshold):
        return NO_ANSWER, hits, False
    return generate(build_prompt(question, hits)), hits, True


if __name__ == "__main__":
    q = "How do I list all installed packages?"
    response, hits, used_kb = answer(q)

    print(f"Question: {q}\n")
    print("Retrieved (closest first — lower cosine distance = more similar):")
    for text, meta, dist in hits:
        weak = "  (weak)" if dist is not None and dist > RELEVANCE_THRESHOLD else ""
        print(f"  dist={dist:.3f}  {meta.get('url', '')}{weak}")

    tag = "answered from KB (Bedrock)" if used_kb else "relevance gate tripped — refused"
    print(f"\nAnswer ({tag}):\n", response)