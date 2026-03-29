"""
FastMCP server exposing Kubeflow RAG tools for IDE integration.

Enables the "Thin Context" flow from the spec: IDE agents (Cursor, Copilot,
Claude Code) connect to this MCP server to query Kubeflow docs and issues
without running their own retrieval infrastructure.

Runs on a separate port (default 8001) from the main API.
"""
import os
import json
import logging

from fastmcp import FastMCP

from agent.tools.docs_search import search_kubeflow_docs
from agent.tools.issues_search import search_kubeflow_issues

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="kubeflow-docs-agent",
    description="MCP server for querying Kubeflow documentation and GitHub issues",
)


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """
    Search Kubeflow official documentation.

    Use this tool when you need information about Kubeflow concepts,
    installation, configuration, pipelines, KServe, Katib, notebooks,
    or any other Kubeflow component.

    Args:
        query: Natural language search query about Kubeflow
        top_k: Number of results to return (default: 5)

    Returns:
        JSON string with matching documentation chunks and citation URLs
    """
    results = search_kubeflow_docs(query, top_k=top_k)
    return json.dumps(results, indent=2)


@mcp.tool()
def search_issues(query: str, top_k: int = 5) -> str:
    """
    Search Kubeflow GitHub issues and discussions.

    Use this tool when debugging errors, looking for known bugs,
    or finding solutions to problems other users have encountered.

    Args:
        query: Description of the error or issue you're investigating
        top_k: Number of results to return (default: 5)

    Returns:
        JSON string with matching issue context and citation URLs
    """
    results = search_kubeflow_issues(query, top_k=top_k)
    return json.dumps(results, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.getenv("MCP_PORT", "8001"))
    logger.info("Starting MCP server on port %d", port)
    mcp.run(transport="sse", host="0.0.0.0", port=port)
