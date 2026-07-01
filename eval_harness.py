"""
Evaluation harness for the Ubuntu RAG assistant — runs against EITHER backend.

Set BACKEND to "local" (Chroma + Ollama) or "aws" (S3 Vectors + Bedrock).
Everything else — the 30 questions, the relevance gate, the report format — is
identical for both, so the two runs are directly comparable. Outputs are named
per backend (eval_local.* / eval_aws.*) so runs don't overwrite each other.

Records per-question retrieval distance, gate decision, answer, and LATENCY
(wall-clock seconds per question), then writes a CSV + an HTML report.

    python eval_harness.py
"""

import csv
import html
import time
import random
import statistics
import webbrowser
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from rag_core import is_relevant, best_distance, RELEVANCE_THRESHOLD

# --- what to run ---
BACKEND = "local"          # "local" (Chroma+Ollama) or "aws" (S3 Vectors+Bedrock)
GENERATE = True            # False = retrieval-only (fast); True = also generate answers
N_QUESTIONS = 30
MODEL = "qwen3:8b"         # local backend only
SEED = 42

OUT_CSV = f"eval_{BACKEND}.csv"
OUT_HTML = f"eval_{BACKEND}.html"


def make_backend():
    """
    Return (retrieve_fn, answer_fn) for the chosen backend, each taking
    (question) and giving the same shapes:
        retrieve_fn(question) -> hits [(text, metadata, distance), ...]
        answer_fn(question)   -> (answer_text, hits, used_kb)
    """
    if BACKEND == "aws":
        import aws_core
        return (lambda q: aws_core.retrieve(q, 4),
                lambda q: aws_core.answer(q, 4))
    else:
        from rag_core import get_collection, get_llm, retrieve, answer
        col = get_collection()
        llm = get_llm(MODEL)
        return (lambda q: retrieve(col, q, 4),
                lambda q: answer(col, llm, q, 4))


def sample_questions(n):
    """Pull n distinct, reasonable-length questions from the dataset (via
    huggingface_hub + pandas, avoiding the `datasets`/aiohttp SSL crash)."""
    repo = "sedthh/ubuntu_dialogue_qa"
    parquet = next(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet")
    )
    path = hf_hub_download(repo, parquet, repo_type="dataset")
    column = pd.read_parquet(path, columns=["INSTRUCTION"])["INSTRUCTION"].tolist()

    candidates = [q.strip() for q in column
                  if isinstance(q, str) and 20 <= len(q.strip()) <= 300]
    random.Random(SEED).shuffle(candidates)
    seen, picked = set(), []
    for q in candidates:
        if q.lower() not in seen:
            seen.add(q.lower())
            picked.append(q)
        if len(picked) == n:
            break
    return picked


CSS = """
body{font-family:system-ui,'Segoe UI',sans-serif;max-width:840px;margin:24px auto;
padding:0 16px;color:#1a1a1a;background:#fafafa;line-height:1.55}
h1{font-size:22px;margin-bottom:4px}
.summary{background:#fff;border:1px solid #e3e3e0;border-radius:8px;padding:12px 16px;
margin:16px 0 24px;font-size:14px}
.summary code{background:#f1efe8;padding:1px 6px;border-radius:4px}
.card{background:#fff;border:1px solid #e3e3e0;border-left-width:4px;border-radius:8px;
padding:14px 16px;margin:12px 0}
.card.ok{border-left-color:#639922}
.card.refuse{border-left-color:#d08700}
.q{font-weight:600;font-size:15px;margin-bottom:8px}
.meta{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px;
color:#555;margin-bottom:10px}
.badge{font-size:12px;font-weight:600;padding:2px 9px;border-radius:11px;color:#fff}
.badge.ok{background:#639922}
.badge.refuse{background:#d08700}
.answer{white-space:pre-wrap;font-size:14px;background:#fcfcfb;border:1px solid #f0eee8;
border-radius:6px;padding:10px 12px}
.answer.empty{color:#999;font-style:italic;background:#fafafa}
details{margin-top:10px;font-size:13px}
summary{cursor:pointer;color:#555}
.src{border-top:1px solid #eee;padding:8px 0}
.src a{font-size:12px}
.src .snip{color:#666;font-size:12px;margin-top:3px}
.threshold{display:flex;align-items:center;text-align:center;color:#d08700;
font-size:12px;font-weight:600;margin:22px 0}
.threshold:before,.threshold:after{content:"";flex:1;border-top:1px dashed #d08700}
.threshold span{padding:0 12px}
"""


def write_html(records, path, threshold):
    recs = sorted(records, key=lambda r: r["dist"] if r["dist"] is not None else 99)
    answered = sum(1 for r in recs if r["passed"])
    dists = [r["dist"] for r in recs if r["dist"] is not None]
    lats = [r["latency"] for r in recs if r["latency"] is not None]

    out = [f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Ubuntu RAG evaluation — {BACKEND}</title><style>{CSS}</style></head><body>"]
    out.append(f"<h1>Ubuntu RAG evaluation — {BACKEND} backend</h1>")
    summary = (f"<b>{len(recs)}</b> questions · <b>{answered}</b> answered · "
               f"<b>{len(recs) - answered}</b> refused · gate cutoff <code>{threshold}</code>")
    if dists:
        summary += (f"<br>best distance: min <code>{min(dists):.3f}</code> · "
                    f"median <code>{statistics.median(dists):.3f}</code> · "
                    f"max <code>{max(dists):.3f}</code>")
    if lats:
        summary += (f"<br>latency (s): median <code>{statistics.median(lats):.2f}</code> · "
                    f"mean <code>{statistics.mean(lats):.2f}</code> · "
                    f"min <code>{min(lats):.2f}</code> · max <code>{max(lats):.2f}</code>")
    out.append(f"<div class='summary'>{summary}</div>")

    crossed = False
    for i, r in enumerate(recs, 1):
        if not crossed and r["dist"] is not None and r["dist"] > threshold:
            out.append(f"<div class='threshold'><span>relevance cutoff {threshold} — "
                       f"entries below answer, above refuse</span></div>")
            crossed = True
        cls = "ok" if r["passed"] else "refuse"
        d = r["dist"]
        meta = f"<span class='badge {cls}'>{'ANSWERED' if r['passed'] else 'REFUSED'}</span>"
        if d is not None:
            meta += f"<span>distance {d:.3f}</span><span>similarity {1 - d:.3f}</span>"
        if r["latency"] is not None:
            meta += f"<span>latency {r['latency']:.2f}s</span>"
        if r["top_source"]:
            meta += f"<a href='{html.escape(r['top_source'])}'>top source ↗</a>"

        ans = r["answer"].strip()
        ans_html = (f"<div class='answer'>{html.escape(ans)}</div>" if ans
                    else "<div class='answer empty'>(retrieval-only run — set GENERATE=True)</div>")

        srcs = []
        for text, m, dist in r["hits"]:
            url = m.get("url", "")
            snip = html.escape(text.strip().replace("\n", " ")[:300])
            dtxt = f"{dist:.3f}" if dist is not None else "n/a"
            srcs.append(f"<div class='src'><a href='{html.escape(url)}'>{html.escape(url)}</a>"
                        f" · dist {dtxt}<div class='snip'>{snip}...</div></div>")
        details = (f"<details><summary>{len(r['hits'])} retrieved sources</summary>"
                   f"{''.join(srcs)}</details>")

        out.append(f"<div class='card {cls}'><div class='q'>{i}. {html.escape(r['question'])}</div>"
                   f"<div class='meta'>{meta}</div>{ans_html}{details}</div>")

    out.append("</body></html>")
    Path(path).write_text("".join(out), encoding="utf-8")


def main():
    print(f"Backend: {BACKEND}   generate: {GENERATE}")
    retrieve_fn, answer_fn = make_backend()
    questions = sample_questions(N_QUESTIONS)

    # warm-up (untimed) so model-load / first-connection cost doesn't skew Q1
    print("Warming up...")
    try:
        (answer_fn if GENERATE else retrieve_fn)(questions[0])
    except Exception as e:
        print(f"  warm-up note: {e}")

    records = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        if GENERATE:
            ans, hits, passed = answer_fn(q)
        else:
            hits = retrieve_fn(q)
            passed = is_relevant(hits)
            ans = ""
        latency = time.perf_counter() - t0

        bd = best_distance(hits)
        top_url = hits[0][1].get("url", "") if hits else ""
        records.append({"question": q, "dist": bd, "passed": passed,
                        "top_source": top_url, "answer": ans, "hits": hits,
                        "latency": latency})
        print(f"[{i:>2}/{len(questions)}] dist={bd:.3f}  {latency:5.2f}s  "
              f"{'answer' if passed else 'REFUSE'}  {q[:50]}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "best_distance", "gate",
                                          "latency_s", "top_source", "answer"])
        w.writeheader()
        for r in records:
            w.writerow({"question": r["question"],
                        "best_distance": round(r["dist"], 3) if r["dist"] is not None else "",
                        "gate": "answer" if r["passed"] else "REFUSE",
                        "latency_s": round(r["latency"], 2),
                        "top_source": r["top_source"], "answer": r["answer"]})

    write_html(records, OUT_HTML, RELEVANCE_THRESHOLD)

    dists = sorted(r["dist"] for r in records if r["dist"] is not None)
    lats = [r["latency"] for r in records]
    answered = sum(1 for r in records if r["passed"])
    print("\n" + "=" * 52)
    print(f"[{BACKEND}] questions: {len(records)}  answered: {answered}  refused: {len(records)-answered}")
    if dists:
        print(f"best_distance  min={dists[0]:.3f}  median={statistics.median(dists):.3f}  max={dists[-1]:.3f}")
    print(f"latency (s)    median={statistics.median(lats):.2f}  mean={statistics.mean(lats):.2f}  "
          f"min={min(lats):.2f}  max={max(lats):.2f}")
    print(f"\nWrote {OUT_CSV} and {OUT_HTML}")
    webbrowser.open(Path(OUT_HTML).resolve().as_uri())


if __name__ == "__main__":
    main()