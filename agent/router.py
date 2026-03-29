"""
Semantic intent router for classifying user queries.

POC uses keyword heuristics for zero-latency routing.
Production upgrade: replace with an LLM classifier call for nuanced intent detection.
"""
import re
import logging

logger = logging.getLogger(__name__)

# keyword sets derived from actual Kubeflow documentation topics
DOCS_KEYWORDS = {
    "install", "setup", "deploy", "configure", "kubeflow", "kfp", "pipeline",
    "notebook", "katib", "kserve", "serving", "training", "operator", "manifest",
    "component", "dsl", "sdk", "tensorboard", "profile", "namespace", "istio",
    "argo", "tekton", "mpi", "pytorch", "tensorflow", "xgboost", "fairing",
    "metadata", "artifact", "experiment", "run", "recurring", "cron",
    "how do i", "how to", "what is", "getting started", "tutorial", "guide",
    "documentation", "docs", "usage", "example", "yaml", "helm", "kustomize",
}

ISSUES_KEYWORDS = {
    "error", "bug", "issue", "fix", "crash", "fail", "broken", "traceback",
    "exception", "timeout", "permission", "denied", "401", "403", "404", "500",
    "oom", "evicted", "pending", "crashloopbackoff", "imagepullbackoff",
    "not working", "doesn't work", "can't", "cannot", "problem", "debug",
    "troubleshoot", "logs", "stderr", "panic", "segfault",
}

GREETING_KEYWORDS = {
    "hello", "hi", "hey", "howdy", "good morning", "good afternoon",
    "good evening", "what's up", "sup", "greetings", "yo",
}


def classify_intent(query: str) -> str:
    """
    Classify query intent using keyword overlap scoring.

    Returns one of: docs, issues, greeting, out_of_scope
    """
    q = query.lower().strip()

    # greetings are short and match exactly
    if q in GREETING_KEYWORDS or any(q.startswith(g) for g in GREETING_KEYWORDS):
        return "greeting"

    tokens = set(re.findall(r'\w+', q))
    bigrams = set()
    words = q.split()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")

    all_terms = tokens | bigrams

    docs_score = len(all_terms & DOCS_KEYWORDS)
    issues_score = len(all_terms & ISSUES_KEYWORDS)

    # also check substring matches for multi-word patterns
    for kw in DOCS_KEYWORDS:
        if " " in kw and kw in q:
            docs_score += 2
    for kw in ISSUES_KEYWORDS:
        if " " in kw and kw in q:
            issues_score += 2

    if docs_score == 0 and issues_score == 0:
        return "out_of_scope"

    intent = "docs" if docs_score >= issues_score else "issues"
    logger.info("Router: query='%.60s...' -> %s (docs=%d, issues=%d)", q, intent, docs_score, issues_score)
    return intent
