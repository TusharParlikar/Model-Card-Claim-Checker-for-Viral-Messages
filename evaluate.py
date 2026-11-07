"""Measure the pipeline and write results/evaluation.md and results/verdicts_10_messages.md.

Uses whichever LLM .env selects. Run: python evaluate.py
"""
import json
import time
from collections import Counter
from pathlib import Path

from requests.adapters import HTTPAdapter
from sentence_transformers import util
from urllib3.util.retry import Retry

import checker

ROOT = Path(__file__).parent
MESSAGES = json.loads((ROOT / "data" / "messages.json").read_text(encoding="utf-8"))
TUNING = [c for m in MESSAGES for c in m["claims"]]
HELDOUT = json.loads((ROOT / "data" / "heldout.json").read_text(encoding="utf-8"))
LABELS = ["supported", "refuted", "unverified"]
MATCH = 0.6  # cosine similarity at which an extracted claim counts as a gold claim


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows])


def verifier_run(name, gold, sources):
    """Return (summary row, detail section) for one set of claims and one choice of sources."""
    rows, pairs, seconds = [], [], []
    failures = 0
    for g in gold:
        start = time.perf_counter()
        r = checker.verify(g["claim"], sources)
        seconds.append(time.perf_counter() - start)
        pairs.append((g["label"], r["verdict"]))
        failures += bool(r["failed"])
        deciding = r["evidence"][0]["label"] if r["verdict"] != "unverified" else "none"
        rows.append([g["claim"], g["label"], r["verdict"], f"{r['confidence']:.2f}", deciding, "yes" if g["label"] == r["verdict"] else "**NO**"])
        print(name, rows[-1], flush=True)
    right = sum(a == b for a, b in pairs)
    counts = Counter(pairs)
    flips = counts[("supported", "refuted")] + counts[("refuted", "supported")]
    summary = [name, f"{right}/{len(gold)} = {right / len(gold):.0%}", flips, failures, f"{sum(seconds) / len(seconds):.1f}s"]
    confusion = [[f"gold {a}"] + [counts[(a, b)] for b in LABELS] for a in LABELS]
    detail = (
        f"#### {name}\n\n" + table(["", *LABELS], confusion) + "\n\n"
        + table(["Claim", "Gold", "Pipeline", "Conf.", "Decided by", "Correct"], rows)
    )
    return summary, detail


def verifier_section():
    both = tuple(checker.SOURCES)
    runs = [
        ("Tuning set, Wikipedia only", TUNING, ("Wikipedia",)),
        ("Tuning set, Wikipedia and news", TUNING, both),
        ("Held-out set, Wikipedia only", HELDOUT, ("Wikipedia",)),
        ("Held-out set, Wikipedia and news", HELDOUT, both),
    ]
    results = [verifier_run(*run) for run in runs]
    return (
        "The tuning set was used to choose thresholds, so its numbers are optimistic. The held-out set was not.\n\n"
        + table(["Run", "Accuracy", "Supported/refuted flipped", "Claims with a source down", "Mean time per claim"], [r[0] for r in results])
        + "\n\n" + "\n\n".join(r[1] for r in results)
    )


def prompt_section():
    rows = []
    for technique in checker.PROMPTS:
        found, matched, recalled, dropped, seconds = 0, 0, 0, 0, 0.0
        for m in MESSAGES:
            start = time.perf_counter()
            kept, gone = checker.extract_claims(m["text"], technique)
            seconds += time.perf_counter() - start
            found += len(kept)
            dropped += len(gone)
            if kept:
                sims = util.cos_sim(checker.embedder().encode(kept), checker.embedder().encode([c["claim"] for c in m["claims"]]))
                matched += int((sims.max(dim=1).values >= MATCH).sum())
                recalled += int((sims.max(dim=0).values >= MATCH).sum())
            print(technique, m["id"], kept, gone, flush=True)
        precision = matched / found if found else 0
        rows.append([technique, found, f"{precision:.0%}", f"{recalled / len(TUNING):.0%}", dropped, f"{seconds / len(MESSAGES):.1f}s"])
    return table(["Prompt", "Claims kept", "Precision", "Recall", "Dropped by grounding check", "Latency per message"], rows)


CLOSED_BOOK = 'From memory, with no sources, is this claim true? Respond only with JSON: {"verdict": "supported" or "refuted" or "unverified"}'


def closed_book_section():
    """Hallucination test: how often does the LLM answer confidently and wrongly with no evidence?"""
    right, wrong_confident, rows = 0, 0, []
    for g in TUNING:
        verdict = str(checker.chat_json(CLOSED_BOOK, g["claim"]).get("verdict", "unverified")).lower()
        right += verdict == g["label"]
        wrong_confident += verdict != g["label"] and verdict in ("supported", "refuted")
        rows.append([g["claim"], g["label"], verdict])
        print(rows[-1], flush=True)
    n = len(TUNING)
    return (
        f"LLM alone, no evidence: **{right}/{n} = {right / n:.0%}** correct, "
        f"**{wrong_confident}/{n} = {wrong_confident / n:.0%}** answered confidently and wrongly.\n\n"
        + table(["Claim", "Gold", "LLM alone"], rows)
    )


def verdicts_section():
    """The P54 deliverable: end-to-end verdicts for 10 messages."""
    out = []
    for m in MESSAGES:
        out.append(f"### Message {m['id']}\n\n> {m['text']}\n")
        claims, dropped = checker.extract_claims(m["text"])
        rows = []
        for claim in claims:
            r = checker.verify(claim)
            top = r["evidence"][0] if r["verdict"] != "unverified" else None
            link = f"[{top['label']}]({top['url']})" if top else "none"
            rows.append([claim, r["verdict"], f"{r['confidence']:.2f}", link])
            print(rows[-1], flush=True)
        out.append(table(["Extracted claim", "Verdict", "Conf.", "Evidence"], rows) if rows else "No claims extracted.")
        if dropped:
            out.append("\nDropped by grounding check: " + "; ".join(dropped))
        out.append("")
    return "\n".join(out)


def main():
    # Unlike a live demo, a measurement can afford to wait out rate limits rather than lose a source.
    checker.HTTP.mount("https://", HTTPAdapter(max_retries=Retry(total=8, backoff_factor=5, status_forcelist=[429, 502, 503])))
    (ROOT / "results").mkdir(exist_ok=True)
    cfg = checker.llm_config()
    source = f"LLM: `{cfg['provider']}` / `{cfg['model']}`. Generated by running `python evaluate.py`."
    verifier = verifier_section()
    prompts = prompt_section()
    closed = closed_book_section()
    verdicts = verdicts_section()
    (ROOT / "results" / "verdicts_10_messages.md").write_text(f"# Verdicts for 10 messages\n\n{source}\n\n{verdicts}\n", encoding="utf-8")
    (ROOT / "results" / "evaluation.md").write_text(
        f"## Measured results\n\n{source}\n\n### Prompt comparison\n\n{prompts}\n\n"
        f"### Verifier on gold claims\n\n{verifier}\n\n### Hallucination test (LLM alone)\n\n{closed}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
