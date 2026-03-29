"""
LangGraph StateGraph implementing the agentic RAG pipeline.

Flow: START -> router -> [docs_agent | issues_agent | greeting | oos] -> synthesizer -> END

Self-correction: if retrieval returns empty, retry once with a broadened query.
"""
import os
import json
import logging
import httpx

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.router import classify_intent
from agent.tools.docs_search import search_kubeflow_docs
from agent.tools.issues_search import search_kubeflow_issues

logger = logging.getLogger(__name__)

LLM_API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are the Kubeflow Documentation Assistant. Your role is to provide
accurate, well-cited answers about Kubeflow based strictly on the retrieved documentation.

Rules:
- Ground every claim in the provided context. Do not hallucinate.
- Include citation URLs in your response where applicable.
- If the context does not contain enough information, say so clearly.
- Be concise but thorough. Prefer code examples when relevant.
- Never expose raw tool call JSON or internal state to the user."""


def _call_llm(messages: list[dict]) -> str:
    """Call the OpenAI-compatible LLM endpoint."""
    if not LLM_API_KEY:
        return _fallback_synthesis(messages)

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    try:
        resp = httpx.post(LLM_API_URL, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return _fallback_synthesis(messages)


def _fallback_synthesis(messages: list[dict]) -> str:
    """Template-based fallback when no LLM API key is configured."""
    # extract context from the last user message
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return "I found some relevant documentation but couldn't generate a synthesis. Please review the citations below."


# --- Graph Nodes ---

def route_node(state: AgentState) -> AgentState:
    """Classify the user query intent."""
    intent = classify_intent(state["query"])
    state["intent"] = intent
    state["retry_count"] = 0
    state["tool_calls"] = []
    logger.info("Routed to: %s", intent)
    return state


def docs_agent_node(state: AgentState) -> AgentState:
    """Search docs_rag and populate results."""
    query = state["query"]
    results = search_kubeflow_docs(query)
    state["search_results"] = results
    state["tool_calls"].append({"tool": "search_kubeflow_docs", "query": query, "hits": len(results)})

    # self-correction: broaden query on empty results
    if not results and state["retry_count"] < 1:
        broader = " ".join(query.split()[:3])  # use first 3 words as broader query
        logger.info("Self-correction: retrying with broader query '%s'", broader)
        results = search_kubeflow_docs(broader, top_k=8)
        state["search_results"] = results
        state["retry_count"] += 1
        state["tool_calls"].append({"tool": "search_kubeflow_docs", "query": broader, "hits": len(results), "retry": True})

    state["citations"] = [r["citation_url"] for r in results if r.get("citation_url")]
    return state


def issues_agent_node(state: AgentState) -> AgentState:
    """Search issues_rag and populate results."""
    query = state["query"]
    results = search_kubeflow_issues(query)
    state["search_results"] = results
    state["tool_calls"].append({"tool": "search_kubeflow_issues", "query": query, "hits": len(results)})

    if not results and state["retry_count"] < 1:
        broader = " ".join(query.split()[:3])
        logger.info("Self-correction: retrying issues search with '%s'", broader)
        results = search_kubeflow_issues(broader, top_k=8)
        state["search_results"] = results
        state["retry_count"] += 1
        state["tool_calls"].append({"tool": "search_kubeflow_issues", "query": broader, "hits": len(results), "retry": True})

    state["citations"] = [r["citation_url"] for r in results if r.get("citation_url")]
    return state


def greeting_node(state: AgentState) -> AgentState:
    """Handle greeting queries without tool calls."""
    state["answer"] = (
        "Hello! I'm the Kubeflow Documentation Assistant. "
        "I can help you with Kubeflow installation, pipelines, KServe, Katib, notebooks, and more. "
        "What would you like to know?"
    )
    state["search_results"] = []
    state["citations"] = []
    return state


def oos_node(state: AgentState) -> AgentState:
    """Handle out-of-scope queries."""
    state["answer"] = (
        "That question seems outside my area of expertise. "
        "I specialize in Kubeflow documentation, including topics like Kubeflow Pipelines, "
        "KServe, Katib, Notebooks, and deployment. Feel free to ask me about any of those!"
    )
    state["search_results"] = []
    state["citations"] = []
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """Use LLM to synthesize a response from retrieved context."""
    if state.get("answer"):
        # greeting or oos already set the answer
        return state

    results = state.get("search_results", [])
    if not results:
        state["answer"] = (
            "I wasn't able to find relevant documentation for your query. "
            "Could you rephrase or provide more detail?"
        )
        return state

    # build context block from retrieved chunks
    context_parts = []
    for i, r in enumerate(results):
        url = r.get("citation_url", "N/A")
        text = r.get("content_text", "")
        context_parts.append(f"[Source {i+1}] ({url})\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"},
    ]

    state["answer"] = _call_llm(messages)
    return state


def _route_by_intent(state: AgentState) -> str:
    """Conditional edge: dispatch to the correct agent node based on intent."""
    return state["intent"]


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph agent pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("router", route_node)
    graph.add_node("docs_agent", docs_agent_node)
    graph.add_node("issues_agent", issues_agent_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("out_of_scope", oos_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "docs": "docs_agent",
            "issues": "issues_agent",
            "greeting": "greeting",
            "out_of_scope": "out_of_scope",
        },
    )

    # all agent nodes converge to synthesizer
    graph.add_edge("docs_agent", "synthesizer")
    graph.add_edge("issues_agent", "synthesizer")
    graph.add_edge("greeting", "synthesizer")
    graph.add_edge("out_of_scope", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
