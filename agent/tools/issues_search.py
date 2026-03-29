"""
MCP tool: search the issues_rag collection for GitHub issues context.

Same singleton and no-truncation pattern as docs_search.
"""
import os
import logging

from agent.tools.base import get_model, get_client

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("ISSUES_COLLECTION", "issues_rag")
CONTENT_MAX_CHARS = int(os.getenv("CONTENT_MAX_CHARS", "0"))


def search_kubeflow_issues(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and search issues_rag for relevant GitHub issues/discussions.

    Returns a list of dicts with content_text, citation_url, file_path, score.
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
        if CONTENT_MAX_CHARS > 0:
            content = content[:CONTENT_MAX_CHARS]

        hits.append({
            "content_text": content,
            "citation_url": hit["entity"].get("citation_url", ""),
            "file_path": hit["entity"].get("file_path", ""),
            "score": round(hit["distance"], 4),
        })

    logger.info("issues_search returned %d results for query: %.60s...", len(hits), query)
    return hits
