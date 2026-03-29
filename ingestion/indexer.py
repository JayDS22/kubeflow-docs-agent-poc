"""
Milvus indexer using pymilvus MilvusClient directly.

Fixes Issue #182: no Feast dependency, no VARCHAR monkey-patch.
Uses upsert keyed on file_unique_id for idempotent writes.
Does NOT drop and recreate collections on each run.
"""
import os
import json
import logging
import time

from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

logger = logging.getLogger(__name__)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "100"))

# schema matches the spec: content_text at 4096 chars (not 512, not 2000)
SCHEMA_FIELDS = [
    FieldSchema("file_unique_id", DataType.VARCHAR, max_length=512, is_primary=True),
    FieldSchema("file_path", DataType.VARCHAR, max_length=512),
    FieldSchema("citation_url", DataType.VARCHAR, max_length=1024),
    FieldSchema("content_text", DataType.VARCHAR, max_length=4096),
    FieldSchema("chunk_index", DataType.INT64),
    FieldSchema("source_type", DataType.VARCHAR, max_length=32),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]


def _wait_for_milvus(client: MilvusClient, max_wait: int = 120) -> None:
    """Block until Milvus is reachable. Used during container startup."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            client.list_collections()
            logger.info("Milvus is ready")
            return
        except Exception:
            logger.info("Waiting for Milvus...")
            time.sleep(3)
    raise RuntimeError(f"Milvus not reachable after {max_wait}s")


def _ensure_collection(client: MilvusClient, name: str) -> None:
    """Create collection if it doesn't exist. Never drop existing data."""
    existing = client.list_collections()
    if name in existing:
        logger.info("Collection '%s' already exists, skipping creation", name)
        return

    schema = CollectionSchema(fields=SCHEMA_FIELDS, description=f"RAG collection: {name}")
    client.create_collection(collection_name=name, schema=schema)

    # create IVF_FLAT index for ANN search
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(collection_name=name, index_params=index_params)
    logger.info("Created collection '%s' with IVF_FLAT COSINE index", name)


def index_documents(input_path: str, collection_name: str = "docs_rag") -> int:
    """
    Read embedded JSONL and upsert into Milvus.

    Uses upsert keyed on file_unique_id for idempotent ingestion.
    Returns the number of records upserted.
    """
    client = MilvusClient(uri=MILVUS_URI)
    _wait_for_milvus(client)
    _ensure_collection(client, collection_name)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # truncate content_text to schema max if needed
            content = rec["content_text"][:4090]
            records.append({
                "file_unique_id": rec["file_unique_id"][:510],
                "file_path": rec["file_path"][:510],
                "citation_url": rec.get("citation_url", "")[:1020],
                "content_text": content,
                "chunk_index": rec["chunk_index"],
                "source_type": rec.get("source_type", "docs")[:30],
                "embedding": rec["embedding"],
            })

    logger.info("Upserting %d records into '%s' in batches of %d", len(records), collection_name, UPSERT_BATCH)

    total = 0
    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i:i + UPSERT_BATCH]
        try:
            client.upsert(collection_name=collection_name, data=batch)
            total += len(batch)
            if total % 500 == 0:
                logger.info("Upserted %d/%d records", total, len(records))
        except Exception as e:
            logger.error("Upsert failed at batch %d: %s", i, e)
            # continue with next batch rather than aborting the entire run
            continue

    # load collection for searching
    client.load_collection(collection_name)
    logger.info("Indexing complete: %d records in '%s'", total, collection_name)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    index_documents("data/embedded_docs.jsonl")
