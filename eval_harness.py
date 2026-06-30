"""
Evaluation harness for the Ask Ubuntu RAG.

Runs a batch of real questions (from the Ubuntu dialogue dataset) through the
pipeline, records each one's best retrieval distance and whether the relevance
gate let it answer, and writes BOTH:
  * eval_results.csv   — flat, sortable in a spreadsheet
  * eval_report.html   — one readable card per question, opens in your browser

Two uses:
  * Threshold tuning (fast): leave GENERATE = False. You only need distances.
  * Quality review (slow): set GENERATE = True to also generate answers to read.

    python eval_harness.py
"""

import csv
import html
import random
import statistics
import webbrowser
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from rag_core import (
    get_collection, get_llm, retrieve, is_relevant, build_prompt,
    best_distance, RELEVANCE_THRESHOLD,
)

N_QUESTIONS = 30
GENERATE = True          # True = also generate answers (slower, for quality review)
MODEL = "qwen3:8b"
OUT_CSV = "eval_results.csv"
OUT_HTML = "eval_report.html"
SEED = 42


def sample_questions(n):
    """
    Pull n distinct, reasonable-length questions from the dataset.

    Avoids the `datasets` library (whose aiohttp import crashes on some
    Windows setups with an SSL/ASN1 error). Downloads the parquet via
    huggingface_hub, which uses requests + certifi, not the Windows cert store.
    """
    repo = "sedthh/ubuntu_dialogue_qa"
    parquet = next(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet")
    )
    path = hf_hub_download(repo, parquet, repo_type="dataset")
    column = pd.read_parquet(path, columns=["INSTRUCTION"])["INSTRUCTION"].tolist()

    candidates = [
        q.strip() for q in column
        if isinstance(q, str) and 20 <= len(q.strip()) <= 300
    ]
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
    recs = sorted(
        records, key=lambda r: r["dist"] if r["dist"] is not None else 99
    )
    answered = sum(1 for r in recs if r["passed"])
    dists = [r["dist"] for r in recs if r["dist"] is not None]

    out = [f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Ask Ubuntu RAG — evaluation</title><style>{CSS}</style></head><body>"]
    out.append("<h1>Ask Ubuntu RAG — evaluation report</h1>")
    summary = (
        f"<b>{len(recs)}</b> questions · <b>{answered}</b> answered · "
        f"<b>{len(recs) - answered}</b> refused · gate cutoff <code>{threshold}</code>"
    )
    if dists:
        summary += (f"<br>best distance: min <code>{min(dists):.3f}</code> · "
                    f"median <code>{statistics.median(dists):.3f}</code> · "
                    f"max <code>{max(dists):.3f}</code>"
                    f"<br><small>Sorted best→worst. Read the cards around the dashed "
                    f"cutoff line to judge whether the threshold sits right.</small>")
    out.append(f"<div class='summary'>{summary}</div>")

    crossed = False
    for i, r in enumerate(recs, 1):
        if (not crossed and r["dist"] is not None and r["dist"] > threshold):
            out.append(f"<div class='threshold'><span>relevance cutoff "
                       f"{threshold} — entries below answer, above refuse</span></div>")
            crossed = True

        cls = "ok" if r["passed"] else "refuse"
        badge = "ANSWERED" if r["passed"] else "REFUSED"
        d = r["dist"]
        meta = f"<span class='badge {cls}'>{badge}</span>"
        if d is not None:
            meta += (f"<span>distance {d:.3f}</span>"
                     f"<span>similarity {1 - d:.3f}</span>")
        if r["top_source"]:
            meta += f"<a href='{html.escape(r['top_source'])}'>top source ↗</a>"

        ans = r["answer"].strip()
        if ans:
            ans_html = f"<div class='answer'>{html.escape(ans)}</div>"
        else:
            ans_html = ("<div class='answer empty'>(retrieval-only run — set "
                        "GENERATE=True to produce answers)</div>")

        srcs = []
        for text, m, dist in r["hits"]:
            url = m.get("url", "")
            snip = html.escape(text.strip().replace("\n", " ")[:300])
            srcs.append(
                f"<div class='src'><a href='{html.escape(url)}'>{html.escape(url)}</a>"
                f" · dist {dist:.3f}<div class='snip'>{snip}...</div></div>"
            )
        details = (f"<details><summary>{len(r['hits'])} retrieved sources</summary>"
                   f"{''.join(srcs)}</details>")

        out.append(
            f"<div class='card {cls}'><div class='q'>{i}. {html.escape(r['question'])}</div>"
            f"<div class='meta'>{meta}</div>{ans_html}{details}</div>"
        )

    out.append("</body></html>")
    Path(path).write_text("".join(out), encoding="utf-8")


def main():
    col = get_collection()
    llm = get_llm(MODEL) if GENERATE else None
    questions = sample_questions(N_QUESTIONS)

    records = []
    for i, q in enumerate(questions, 1):
        hits = retrieve(col, q, k=4)
        bd = best_distance(hits)
        passed = is_relevant(hits)               # uses default threshold
        top_url = hits[0][1].get("url", "") if hits else ""

        ans = ""
        if GENERATE:
            ans = (llm.invoke(build_prompt(q, hits)).content
                   if passed else "[gated — refused]")

        records.append({
            "question": q, "dist": bd, "passed": passed,
            "top_source": top_url, "answer": ans, "hits": hits,
        })
        print(f"[{i:>2}/{len(questions)}] dist={bd:.3f}  "
              f"{'answer' if passed else 'REFUSE'}  {q[:60]}")

    # flat CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "best_distance", "gate", "top_source", "answer"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "question": r["question"],
                "best_distance": round(r["dist"], 3) if r["dist"] is not None else "",
                "gate": "answer" if r["passed"] else "REFUSE",
                "top_source": r["top_source"],
                "answer": r["answer"],
            })

    # readable HTML report
    write_html(records, OUT_HTML, RELEVANCE_THRESHOLD)

    dists = sorted(r["dist"] for r in records if r["dist"] is not None)
    answered = sum(1 for r in records if r["passed"])
    print("\n" + "=" * 52)
    print(f"Questions: {len(records)}   answered: {answered}   refused: {len(records) - answered}")
    if dists:
        print(f"best_distance  min={dists[0]:.3f}  "
              f"median={statistics.median(dists):.3f}  max={dists[-1]:.3f}")
        print("sorted distances:", ", ".join(f"{d:.2f}" for d in dists))
    print(f"current gate threshold: {RELEVANCE_THRESHOLD}")
    print(f"\nWrote {OUT_CSV} and {OUT_HTML}")

    # open the report in the default browser
    webbrowser.open(Path(OUT_HTML).resolve().as_uri())


if __name__ == "__main__":
    main()