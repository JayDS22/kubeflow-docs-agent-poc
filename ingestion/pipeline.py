"""
Ingestion pipeline: scrape -> chunk -> embed -> index.

Orchestrates all stages with timing and error reporting.
Designed to run as a one-shot container in Docker Compose or as a KFP pipeline step.
"""
import os
import sys
import time
import logging

from ingestion.scraper import scrape_docs
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_chunks
from ingestion.indexer import index_documents

logger = logging.getLogger(__name__)

COLLECTION = os.getenv("DOCS_COLLECTION", "docs_rag")


def run_pipeline() -> None:
    """Execute the full ingestion pipeline with stage timing."""
    start = time.time()
    logger.info("=== Starting ingestion pipeline ===")

    # stage 1: scrape
    t0 = time.time()
    raw_path = scrape_docs("data/raw_docs.jsonl")
    logger.info("Stage 1 (scrape): %.1fs", time.time() - t0)

    # stage 2: chunk
    t0 = time.time()
    chunked_path = chunk_documents(raw_path, "data/chunked_docs.jsonl")
    logger.info("Stage 2 (chunk): %.1fs", time.time() - t0)

    # stage 3: embed
    t0 = time.time()
    embedded_path = embed_chunks(chunked_path, "data/embedded_docs.jsonl")
    logger.info("Stage 3 (embed): %.1fs", time.time() - t0)

    # stage 4: index
    t0 = time.time()
    count = index_documents(embedded_path, COLLECTION)
    logger.info("Stage 4 (index): %.1fs", time.time() - t0)

    total = time.time() - start
    logger.info("=== Pipeline complete: %d chunks indexed in %.1fs ===", count, total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run_pipeline()
