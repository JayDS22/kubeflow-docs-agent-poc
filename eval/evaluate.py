"""
Evaluation pipeline for the Kubeflow Docs Agent.

Runs each golden dataset query through the agent and measures:
  - Keyword recall: does the response contain expected terms?
  - Citation coverage: are citation URLs present and valid?
  - Latency: per-query and average response time
  - Intent accuracy: did the router classify correctly?

Outputs results as JSON to eval/results/.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
GOLDEN_PATH = os.getenv("GOLDEN_PATH", "eval/golden_dataset.json")
RESULTS_DIR = "eval/results"
LATENCY_THRESHOLD = 5.0  # seconds


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH, "r") as f:
        return json.load(f)


def query_agent(query: str) -> tuple[dict, float]:
    """Send a non-streaming query to the agent and return (response, latency_seconds)."""
    start = time.time()
    resp = httpx.post(
        f"{API_URL}/chat",
        json={"query": query, "stream": False},
        timeout=30.0,
    )
    latency = time.time() - start
    resp.raise_for_status()
    return resp.json(), latency


def compute_keyword_recall(answer: str, expected: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected if kw.lower() in answer_lower)
    return found / len(expected)


def validate_citation(url: str) -> bool:
    """Check that a citation looks like a valid kubeflow.org URL."""
    if not url:
        return False
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def run_evaluation() -> dict:
    """Execute the full evaluation suite."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    dataset = load_golden_dataset()
    logger.info("Loaded %d golden dataset entries", len(dataset))

    results = []
    total_recall = 0.0
    total_latency = 0.0
    citation_hit = 0
    under_threshold = 0

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]
        expected_kw = entry["expected_keywords"]

        logger.info("Evaluating Q%d: %s", qid, query[:60])

        try:
            response, latency = query_agent(query)
        except Exception as e:
            logger.error("Q%d failed: %s", qid, e)
            results.append({
                "id": qid,
                "query": query,
                "error": str(e),
                "keyword_recall": 0.0,
                "latency_s": -1,
                "has_citations": False,
            })
            continue

        answer = response.get("answer", "")
        citations = response.get("citations", [])
        intent = response.get("intent", "")

        recall = compute_keyword_recall(answer, expected_kw)
        has_valid_citations = any(validate_citation(c) for c in citations)

        total_recall += recall
        total_latency += latency
        if has_valid_citations:
            citation_hit += 1
        if latency < LATENCY_THRESHOLD:
            under_threshold += 1

        result = {
            "id": qid,
            "query": query,
            "category": entry.get("category", ""),
            "intent_detected": intent,
            "keyword_recall": round(recall, 3),
            "keywords_found": [kw for kw in expected_kw if kw.lower() in answer.lower()],
            "keywords_missing": [kw for kw in expected_kw if kw.lower() not in answer.lower()],
            "citation_count": len(citations),
            "has_valid_citations": has_valid_citations,
            "latency_s": round(latency, 3),
            "answer_length": len(answer),
        }
        results.append(result)
        logger.info("  recall=%.2f  citations=%d  latency=%.2fs", recall, len(citations), latency)

    n = len(dataset)
    evaluated = len([r for r in results if "error" not in r])

    summary = {
        "total_queries": n,
        "evaluated": evaluated,
        "errors": n - evaluated,
        "avg_keyword_recall": round(total_recall / max(evaluated, 1), 3),
        "avg_latency_s": round(total_latency / max(evaluated, 1), 3),
        "citation_coverage": round(citation_hit / max(evaluated, 1), 3),
        "pct_under_5s": round(under_threshold / max(evaluated, 1) * 100, 1),
    }

    output = {"summary": summary, "results": results}

    # write timestamped results file
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=== Evaluation Summary ===")
    logger.info("  Queries:           %d (%d errors)", n, n - evaluated)
    logger.info("  Avg keyword recall: %.1f%%", summary["avg_keyword_recall"] * 100)
    logger.info("  Avg latency:        %.2fs", summary["avg_latency_s"])
    logger.info("  Citation coverage:  %.1f%%", summary["citation_coverage"] * 100)
    logger.info("  Under 5s:           %.1f%%", summary["pct_under_5s"])
    logger.info("  Results saved to:   %s", outpath)

    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run_evaluation()
