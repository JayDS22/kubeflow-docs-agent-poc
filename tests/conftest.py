"""
Shared test fixtures. Patches Milvus and embedding model for unit tests
so tests run without external dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_milvus_client():
    """Mocked MilvusClient that returns canned search results."""
    client = MagicMock()
    client.list_collections.return_value = ["docs_rag", "issues_rag"]
    client.search.return_value = [[
        {
            "id": 1,
            "distance": 0.92,
            "entity": {
                "content_text": "Install Kubeflow using kustomize on a local kind cluster.",
                "citation_url": "https://www.kubeflow.org/docs/started/installing/",
                "file_path": "content/en/docs/started/installing.md",
                "chunk_index": 0,
            },
        },
        {
            "id": 2,
            "distance": 0.85,
            "entity": {
                "content_text": "KServe provides serverless inference on Kubernetes.",
                "citation_url": "https://www.kubeflow.org/docs/external-add-ons/kserve/",
                "file_path": "content/en/docs/external-add-ons/kserve/kserve.md",
                "chunk_index": 0,
            },
        },
    ]]
    return client


@pytest.fixture
def mock_embedding_model():
    """Mocked SentenceTransformer that returns deterministic 768-dim vectors."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture(autouse=True)
def patch_singletons(mock_milvus_client, mock_embedding_model):
    """Inject mocks into the singleton module so no real connections are made."""
    with patch("agent.tools.base._client", mock_milvus_client), \
         patch("agent.tools.base._model", mock_embedding_model):
        yield
