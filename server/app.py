"""
FastAPI server exposing the agentic RAG pipeline.

Endpoints:
  POST /chat         SSE streaming (default) or JSON response (stream=false)
  GET  /ws           WebSocket for real-time chat
  GET  /health       Checks Milvus connection + model readiness (not static "OK")

Addresses health check improvement from PR #53.
"""
import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.graph import build_graph
from agent.tools.base import check_milvus_health, check_model_health

logger = logging.getLogger(__name__)

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the LangGraph agent once at startup."""
    global graph
    logger.info("Compiling LangGraph agent pipeline")
    graph = build_graph()
    logger.info("Agent pipeline ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Kubeflow Docs Agent",
    description="Agentic RAG for Kubeflow documentation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    stream: bool = Field(default=True)


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    intent: str
    tool_calls: list


def _run_agent(query: str) -> dict:
    """Execute the LangGraph pipeline synchronously and return state."""
    initial_state = {
        "query": query,
        "intent": "",
        "search_results": [],
        "answer": "",
        "citations": [],
        "tool_calls": [],
        "retry_count": 0,
    }
    result = graph.invoke(initial_state)
    return result


@app.get("/health")
async def health():
    """
    Health check that verifies actual dependency readiness.
    Returns component-level status instead of a static 200.
    """
    milvus_ok = check_milvus_health()
    model_ok = check_model_health()
    healthy = milvus_ok and model_ok

    status = {
        "status": "healthy" if healthy else "degraded",
        "milvus_connected": milvus_ok,
        "model_loaded": model_ok,
    }

    return JSONResponse(
        content=status,
        status_code=200 if healthy else 503,
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint. Returns SSE stream by default, JSON if stream=false.
    """
    if not req.stream:
        result = await asyncio.to_thread(_run_agent, req.query)
        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            intent=result["intent"],
            tool_calls=result["tool_calls"],
        )

    # SSE streaming: emit the answer in chunks for real-time display
    async def event_stream():
        result = await asyncio.to_thread(_run_agent, req.query)

        # stream the answer in word-level chunks for smooth UX
        words = result["answer"].split(" ")
        buffer = ""
        for i, word in enumerate(words):
            buffer += word + " "
            if i % 3 == 2 or i == len(words) - 1:
                data = json.dumps({"type": "token", "content": buffer})
                yield f"data: {data}\n\n"
                buffer = ""
                await asyncio.sleep(0.02)

        # send citations and metadata as final event
        meta = json.dumps({
            "type": "done",
            "citations": result["citations"],
            "intent": result["intent"],
            "tool_calls": result["tool_calls"],
        })
        yield f"data: {meta}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    """WebSocket endpoint for real-time bidirectional chat."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                query = msg.get("query", data)
            except json.JSONDecodeError:
                query = data

            result = await asyncio.to_thread(_run_agent, query)
            await ws.send_json({
                "answer": result["answer"],
                "citations": result["citations"],
                "intent": result["intent"],
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
