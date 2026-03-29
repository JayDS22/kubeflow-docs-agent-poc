"""
LangGraph state schema for the agentic RAG pipeline.
"""
from typing import TypedDict


class AgentState(TypedDict):
    query: str
    intent: str              # docs, issues, greeting, out_of_scope
    search_results: list     # raw results from MCP tools
    answer: str              # synthesized response
    citations: list[str]     # citation URLs extracted from search results
    tool_calls: list         # trace of tool invocations for observability
    retry_count: int         # self-correction loop counter
