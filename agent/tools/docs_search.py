"""
MCP tool: search the docs_rag collection for Kubeflow documentation.

Fixes Issue #181: content is NOT truncated before returning to the agent.
Truncation limit is configurable via CONTENT_MAX_CHARS (default=0, disabled).
"""
import os
import logging

from agent.tools.base import get_model, get_client

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("DOCS_COLLECTION", "docs_rag")
CONTENT_MAX_CHARS = int(os.getenv("CONTENT_MAX_CHARS", "0"))


def search_kubeflow_docs(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and search docs_rag for relevant documentation chunks.

    Returns a list of dicts with content_text, citation_url, file_path, score.
    Content is returned untruncated by default (CONTENT_MAX_CHARS=0).
    """
    model = get_model()
    client = get_client()

    query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

    try:
        results = client.search(
            collection_name=COLLECTION,
            data=[query_embedding],
            limit=top_k,
            output_fields=["content_text", "citation_url", "file_path", "chunk_index"],
            search_params={"metric_type": "COSINE"},
        )
    except Exception as e:
        logger.error("Milvus search failed on %s: %s", COLLECTION, e)
        return []

    hits = []
    for hit in results[0]:
        content = hit["entity"].get("content_text", "")
        # apply truncation only if explicitly configured (fixes Issue #181)
        if CONTENT_MAX_CHARS > 0:
            content = content[:CONTENT_MAX_CHARS]

        hits.append({
            "content_text": content,
            "citation_url": hit["entity"].get("citation_url", ""),
            "file_path": hit["entity"].get("file_path", ""),
            "score": round(hit["distance"], 4),
        })

    logger.info("docs_search returned %d results for query: %.60s...", len(hits), query)
    return hits
