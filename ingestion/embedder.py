"""
Embedding generator using SentenceTransformer.

Uses singleton pattern to load the model exactly once.
Fixes Issue #183: eliminates repeated ~3s model load per batch.
"""
import os
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# singleton: load once at module level on first import
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s (singleton init)", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded: dim=%d", _model.get_sentence_embedding_dimension())
    return _model


def embed_chunks(input_path: str, output_path: str = "data/embedded_docs.jsonl") -> str:
    """
    Read chunked JSONL, generate embeddings, and write enriched JSONL.

    Batches encoding for throughput. Model is loaded once via singleton.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = _get_model()

    # read all chunks first for batched encoding
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info("Embedding %d chunks in batches of %d", len(records), BATCH_SIZE)
    texts = [r["content_text"] for r in records]

    # batch encode
    all_embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record, embedding in zip(records, all_embeddings):
            record["embedding"] = embedding.tolist()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Embedding complete: %d vectors written to %s", len(records), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    embed_chunks("data/chunked_docs.jsonl")
