"""
Centralized configuration via environment variables.
No hardcoded credentials. Every tunable exposed as an env var with sane defaults.
"""
import os


# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Collections
DOCS_COLLECTION = os.getenv("DOCS_COLLECTION", "docs_rag")
ISSUES_COLLECTION = os.getenv("ISSUES_COLLECTION", "issues_rag")

# LLM backend (OpenAI-compatible)
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Content truncation: 0 = no truncation (fixes Issue #181)
CONTENT_MAX_CHARS = int(os.getenv("CONTENT_MAX_CHARS", "0"))

# Ingestion
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Search defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
