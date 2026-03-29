"""
Shared singletons for SentenceTransformer and MilvusClient.

Fixes Issue #183: eliminates ~3s compound initialization cost per request.
Both model and client are loaded once at first access and reused globally.
This is the same pattern used in kagent-feast-mcp/mcp-server/server.py.
"""
import os
import logging

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_client: MilvusClient | None = None


def get_model() -> SentenceTransformer:
    """Return the global embedding model, loading it once on first call."""
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        logger.info("Loading embedding model: %s (singleton init)", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
    return _model


def get_client() -> MilvusClient:
    """Return the global Milvus client, connecting once on first call."""
    global _client
    if _client is None:
        uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        logger.info("Connecting to Milvus at %s (singleton init)", uri)
        _client = MilvusClient(uri=uri)
        logger.info("Milvus connection established")
    return _client


def check_milvus_health() -> bool:
    """Verify Milvus is reachable. Used by the /health endpoint."""
    try:
        client = get_client()
        client.list_collections()
        return True
    except Exception as e:
        logger.error("Milvus health check failed: %s", e)
        return False


def check_model_health() -> bool:
    """Verify the embedding model is loaded and functional."""
    try:
        model = get_model()
        # quick encode to validate the model works
        model.encode(["health check"], show_progress_bar=False)
        return True
    except Exception as e:
        logger.error("Model health check failed: %s", e)
        return False
